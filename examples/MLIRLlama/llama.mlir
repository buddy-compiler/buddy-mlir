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
  func.func @forward(%arg0: tensor<6755192832xf32>, %arg1: tensor<1x80xi64>) -> tensor<1x80x32000xf32> {
    %extracted_slice = tensor.extract_slice %arg0[0] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_0 = tensor.extract_slice %arg0[4096] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_1 = tensor.extract_slice %arg0[8192] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_2 = tensor.extract_slice %arg0[12288] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_3 = tensor.extract_slice %arg0[16384] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_4 = tensor.extract_slice %arg0[20480] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_5 = tensor.extract_slice %arg0[24576] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_6 = tensor.extract_slice %arg0[28672] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_7 = tensor.extract_slice %arg0[32768] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_8 = tensor.extract_slice %arg0[36864] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_9 = tensor.extract_slice %arg0[40960] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_10 = tensor.extract_slice %arg0[45056] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_11 = tensor.extract_slice %arg0[49152] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_12 = tensor.extract_slice %arg0[53248] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_13 = tensor.extract_slice %arg0[57344] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_14 = tensor.extract_slice %arg0[61440] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_15 = tensor.extract_slice %arg0[65536] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_16 = tensor.extract_slice %arg0[69632] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_17 = tensor.extract_slice %arg0[73728] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_18 = tensor.extract_slice %arg0[77824] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_19 = tensor.extract_slice %arg0[81920] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_20 = tensor.extract_slice %arg0[86016] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_21 = tensor.extract_slice %arg0[90112] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_22 = tensor.extract_slice %arg0[94208] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_23 = tensor.extract_slice %arg0[98304] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_24 = tensor.extract_slice %arg0[102400] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_25 = tensor.extract_slice %arg0[106496] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_26 = tensor.extract_slice %arg0[110592] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_27 = tensor.extract_slice %arg0[114688] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_28 = tensor.extract_slice %arg0[118784] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_29 = tensor.extract_slice %arg0[122880] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_30 = tensor.extract_slice %arg0[126976] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_31 = tensor.extract_slice %arg0[131072] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_32 = tensor.extract_slice %arg0[135168] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_33 = tensor.extract_slice %arg0[139264] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_34 = tensor.extract_slice %arg0[143360] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_35 = tensor.extract_slice %arg0[147456] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_36 = tensor.extract_slice %arg0[151552] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_37 = tensor.extract_slice %arg0[155648] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_38 = tensor.extract_slice %arg0[159744] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_39 = tensor.extract_slice %arg0[163840] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_40 = tensor.extract_slice %arg0[167936] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_41 = tensor.extract_slice %arg0[172032] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_42 = tensor.extract_slice %arg0[176128] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_43 = tensor.extract_slice %arg0[180224] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_44 = tensor.extract_slice %arg0[184320] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_45 = tensor.extract_slice %arg0[188416] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_46 = tensor.extract_slice %arg0[192512] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_47 = tensor.extract_slice %arg0[196608] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_48 = tensor.extract_slice %arg0[200704] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_49 = tensor.extract_slice %arg0[204800] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_50 = tensor.extract_slice %arg0[208896] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_51 = tensor.extract_slice %arg0[212992] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_52 = tensor.extract_slice %arg0[217088] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_53 = tensor.extract_slice %arg0[221184] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_54 = tensor.extract_slice %arg0[225280] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_55 = tensor.extract_slice %arg0[229376] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_56 = tensor.extract_slice %arg0[233472] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_57 = tensor.extract_slice %arg0[237568] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_58 = tensor.extract_slice %arg0[241664] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_59 = tensor.extract_slice %arg0[245760] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_60 = tensor.extract_slice %arg0[249856] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_61 = tensor.extract_slice %arg0[253952] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_62 = tensor.extract_slice %arg0[258048] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_63 = tensor.extract_slice %arg0[262144] [4096] [1] : tensor<6755192832xf32> to tensor<4096xf32>
    %extracted_slice_64 = tensor.extract_slice %arg0[266240] [131072000] [1] : tensor<6755192832xf32> to tensor<131072000xf32>
    %expanded = tensor.expand_shape %extracted_slice_64 [[0, 1]] : tensor<131072000xf32> into tensor<32000x4096xf32>
    %extracted_slice_65 = tensor.extract_slice %arg0[131338240] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_66 = tensor.expand_shape %extracted_slice_65 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_67 = tensor.extract_slice %arg0[148115456] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_68 = tensor.expand_shape %extracted_slice_67 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_69 = tensor.extract_slice %arg0[164892672] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_70 = tensor.expand_shape %extracted_slice_69 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_71 = tensor.extract_slice %arg0[181669888] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_72 = tensor.expand_shape %extracted_slice_71 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_73 = tensor.extract_slice %arg0[198447104] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_74 = tensor.expand_shape %extracted_slice_73 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_75 = tensor.extract_slice %arg0[243535872] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_76 = tensor.expand_shape %extracted_slice_75 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_77 = tensor.extract_slice %arg0[288624640] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_78 = tensor.expand_shape %extracted_slice_77 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_79 = tensor.extract_slice %arg0[333713408] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_80 = tensor.expand_shape %extracted_slice_79 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_81 = tensor.extract_slice %arg0[350490624] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_82 = tensor.expand_shape %extracted_slice_81 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_83 = tensor.extract_slice %arg0[367267840] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_84 = tensor.expand_shape %extracted_slice_83 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_85 = tensor.extract_slice %arg0[384045056] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_86 = tensor.expand_shape %extracted_slice_85 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_87 = tensor.extract_slice %arg0[400822272] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_88 = tensor.expand_shape %extracted_slice_87 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_89 = tensor.extract_slice %arg0[445911040] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_90 = tensor.expand_shape %extracted_slice_89 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_91 = tensor.extract_slice %arg0[490999808] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_92 = tensor.expand_shape %extracted_slice_91 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_93 = tensor.extract_slice %arg0[536088576] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_94 = tensor.expand_shape %extracted_slice_93 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_95 = tensor.extract_slice %arg0[552865792] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_96 = tensor.expand_shape %extracted_slice_95 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_97 = tensor.extract_slice %arg0[569643008] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_98 = tensor.expand_shape %extracted_slice_97 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_99 = tensor.extract_slice %arg0[586420224] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_100 = tensor.expand_shape %extracted_slice_99 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_101 = tensor.extract_slice %arg0[603197440] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_102 = tensor.expand_shape %extracted_slice_101 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_103 = tensor.extract_slice %arg0[648286208] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_104 = tensor.expand_shape %extracted_slice_103 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_105 = tensor.extract_slice %arg0[693374976] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_106 = tensor.expand_shape %extracted_slice_105 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_107 = tensor.extract_slice %arg0[738463744] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_108 = tensor.expand_shape %extracted_slice_107 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_109 = tensor.extract_slice %arg0[755240960] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_110 = tensor.expand_shape %extracted_slice_109 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_111 = tensor.extract_slice %arg0[772018176] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_112 = tensor.expand_shape %extracted_slice_111 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_113 = tensor.extract_slice %arg0[788795392] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_114 = tensor.expand_shape %extracted_slice_113 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_115 = tensor.extract_slice %arg0[805572608] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_116 = tensor.expand_shape %extracted_slice_115 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_117 = tensor.extract_slice %arg0[850661376] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_118 = tensor.expand_shape %extracted_slice_117 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_119 = tensor.extract_slice %arg0[895750144] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_120 = tensor.expand_shape %extracted_slice_119 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_121 = tensor.extract_slice %arg0[940838912] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_122 = tensor.expand_shape %extracted_slice_121 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_123 = tensor.extract_slice %arg0[957616128] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_124 = tensor.expand_shape %extracted_slice_123 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_125 = tensor.extract_slice %arg0[974393344] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_126 = tensor.expand_shape %extracted_slice_125 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_127 = tensor.extract_slice %arg0[991170560] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_128 = tensor.expand_shape %extracted_slice_127 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_129 = tensor.extract_slice %arg0[1007947776] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_130 = tensor.expand_shape %extracted_slice_129 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_131 = tensor.extract_slice %arg0[1053036544] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_132 = tensor.expand_shape %extracted_slice_131 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_133 = tensor.extract_slice %arg0[1098125312] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_134 = tensor.expand_shape %extracted_slice_133 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_135 = tensor.extract_slice %arg0[1143214080] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_136 = tensor.expand_shape %extracted_slice_135 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_137 = tensor.extract_slice %arg0[1159991296] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_138 = tensor.expand_shape %extracted_slice_137 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_139 = tensor.extract_slice %arg0[1176768512] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_140 = tensor.expand_shape %extracted_slice_139 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_141 = tensor.extract_slice %arg0[1193545728] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_142 = tensor.expand_shape %extracted_slice_141 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_143 = tensor.extract_slice %arg0[1210322944] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_144 = tensor.expand_shape %extracted_slice_143 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_145 = tensor.extract_slice %arg0[1255411712] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_146 = tensor.expand_shape %extracted_slice_145 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_147 = tensor.extract_slice %arg0[1300500480] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_148 = tensor.expand_shape %extracted_slice_147 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_149 = tensor.extract_slice %arg0[1345589248] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_150 = tensor.expand_shape %extracted_slice_149 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_151 = tensor.extract_slice %arg0[1362366464] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_152 = tensor.expand_shape %extracted_slice_151 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_153 = tensor.extract_slice %arg0[1379143680] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_154 = tensor.expand_shape %extracted_slice_153 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_155 = tensor.extract_slice %arg0[1395920896] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_156 = tensor.expand_shape %extracted_slice_155 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_157 = tensor.extract_slice %arg0[1412698112] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_158 = tensor.expand_shape %extracted_slice_157 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_159 = tensor.extract_slice %arg0[1457786880] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_160 = tensor.expand_shape %extracted_slice_159 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_161 = tensor.extract_slice %arg0[1502875648] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_162 = tensor.expand_shape %extracted_slice_161 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_163 = tensor.extract_slice %arg0[1547964416] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_164 = tensor.expand_shape %extracted_slice_163 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_165 = tensor.extract_slice %arg0[1564741632] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_166 = tensor.expand_shape %extracted_slice_165 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_167 = tensor.extract_slice %arg0[1581518848] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_168 = tensor.expand_shape %extracted_slice_167 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_169 = tensor.extract_slice %arg0[1598296064] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_170 = tensor.expand_shape %extracted_slice_169 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_171 = tensor.extract_slice %arg0[1615073280] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_172 = tensor.expand_shape %extracted_slice_171 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_173 = tensor.extract_slice %arg0[1660162048] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_174 = tensor.expand_shape %extracted_slice_173 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_175 = tensor.extract_slice %arg0[1705250816] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_176 = tensor.expand_shape %extracted_slice_175 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_177 = tensor.extract_slice %arg0[1750339584] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_178 = tensor.expand_shape %extracted_slice_177 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_179 = tensor.extract_slice %arg0[1767116800] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_180 = tensor.expand_shape %extracted_slice_179 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_181 = tensor.extract_slice %arg0[1783894016] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_182 = tensor.expand_shape %extracted_slice_181 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_183 = tensor.extract_slice %arg0[1800671232] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_184 = tensor.expand_shape %extracted_slice_183 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_185 = tensor.extract_slice %arg0[1817448448] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_186 = tensor.expand_shape %extracted_slice_185 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_187 = tensor.extract_slice %arg0[1862537216] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_188 = tensor.expand_shape %extracted_slice_187 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_189 = tensor.extract_slice %arg0[1907625984] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_190 = tensor.expand_shape %extracted_slice_189 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_191 = tensor.extract_slice %arg0[1952714752] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_192 = tensor.expand_shape %extracted_slice_191 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_193 = tensor.extract_slice %arg0[1969491968] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_194 = tensor.expand_shape %extracted_slice_193 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_195 = tensor.extract_slice %arg0[1986269184] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_196 = tensor.expand_shape %extracted_slice_195 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_197 = tensor.extract_slice %arg0[2003046400] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_198 = tensor.expand_shape %extracted_slice_197 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_199 = tensor.extract_slice %arg0[2019823616] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_200 = tensor.expand_shape %extracted_slice_199 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_201 = tensor.extract_slice %arg0[2064912384] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_202 = tensor.expand_shape %extracted_slice_201 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_203 = tensor.extract_slice %arg0[2110001152] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_204 = tensor.expand_shape %extracted_slice_203 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_205 = tensor.extract_slice %arg0[2155089920] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_206 = tensor.expand_shape %extracted_slice_205 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_207 = tensor.extract_slice %arg0[2171867136] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_208 = tensor.expand_shape %extracted_slice_207 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_209 = tensor.extract_slice %arg0[2188644352] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_210 = tensor.expand_shape %extracted_slice_209 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_211 = tensor.extract_slice %arg0[2205421568] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_212 = tensor.expand_shape %extracted_slice_211 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_213 = tensor.extract_slice %arg0[2222198784] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_214 = tensor.expand_shape %extracted_slice_213 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_215 = tensor.extract_slice %arg0[2267287552] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_216 = tensor.expand_shape %extracted_slice_215 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_217 = tensor.extract_slice %arg0[2312376320] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_218 = tensor.expand_shape %extracted_slice_217 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_219 = tensor.extract_slice %arg0[2357465088] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_220 = tensor.expand_shape %extracted_slice_219 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_221 = tensor.extract_slice %arg0[2374242304] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_222 = tensor.expand_shape %extracted_slice_221 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_223 = tensor.extract_slice %arg0[2391019520] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_224 = tensor.expand_shape %extracted_slice_223 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_225 = tensor.extract_slice %arg0[2407796736] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_226 = tensor.expand_shape %extracted_slice_225 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_227 = tensor.extract_slice %arg0[2424573952] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_228 = tensor.expand_shape %extracted_slice_227 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_229 = tensor.extract_slice %arg0[2469662720] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_230 = tensor.expand_shape %extracted_slice_229 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_231 = tensor.extract_slice %arg0[2514751488] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_232 = tensor.expand_shape %extracted_slice_231 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_233 = tensor.extract_slice %arg0[2559840256] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_234 = tensor.expand_shape %extracted_slice_233 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_235 = tensor.extract_slice %arg0[2576617472] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_236 = tensor.expand_shape %extracted_slice_235 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_237 = tensor.extract_slice %arg0[2593394688] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_238 = tensor.expand_shape %extracted_slice_237 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_239 = tensor.extract_slice %arg0[2610171904] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_240 = tensor.expand_shape %extracted_slice_239 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_241 = tensor.extract_slice %arg0[2626949120] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_242 = tensor.expand_shape %extracted_slice_241 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_243 = tensor.extract_slice %arg0[2672037888] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_244 = tensor.expand_shape %extracted_slice_243 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_245 = tensor.extract_slice %arg0[2717126656] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_246 = tensor.expand_shape %extracted_slice_245 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_247 = tensor.extract_slice %arg0[2762215424] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_248 = tensor.expand_shape %extracted_slice_247 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_249 = tensor.extract_slice %arg0[2778992640] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_250 = tensor.expand_shape %extracted_slice_249 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_251 = tensor.extract_slice %arg0[2795769856] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_252 = tensor.expand_shape %extracted_slice_251 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_253 = tensor.extract_slice %arg0[2812547072] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_254 = tensor.expand_shape %extracted_slice_253 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_255 = tensor.extract_slice %arg0[2829324288] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_256 = tensor.expand_shape %extracted_slice_255 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_257 = tensor.extract_slice %arg0[2874413056] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_258 = tensor.expand_shape %extracted_slice_257 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_259 = tensor.extract_slice %arg0[2919501824] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_260 = tensor.expand_shape %extracted_slice_259 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_261 = tensor.extract_slice %arg0[2964590592] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_262 = tensor.expand_shape %extracted_slice_261 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_263 = tensor.extract_slice %arg0[2981367808] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_264 = tensor.expand_shape %extracted_slice_263 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_265 = tensor.extract_slice %arg0[2998145024] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_266 = tensor.expand_shape %extracted_slice_265 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_267 = tensor.extract_slice %arg0[3014922240] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_268 = tensor.expand_shape %extracted_slice_267 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_269 = tensor.extract_slice %arg0[3031699456] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_270 = tensor.expand_shape %extracted_slice_269 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_271 = tensor.extract_slice %arg0[3076788224] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_272 = tensor.expand_shape %extracted_slice_271 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_273 = tensor.extract_slice %arg0[3121876992] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_274 = tensor.expand_shape %extracted_slice_273 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_275 = tensor.extract_slice %arg0[3166965760] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_276 = tensor.expand_shape %extracted_slice_275 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_277 = tensor.extract_slice %arg0[3183742976] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_278 = tensor.expand_shape %extracted_slice_277 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_279 = tensor.extract_slice %arg0[3200520192] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_280 = tensor.expand_shape %extracted_slice_279 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_281 = tensor.extract_slice %arg0[3217297408] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_282 = tensor.expand_shape %extracted_slice_281 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_283 = tensor.extract_slice %arg0[3234074624] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_284 = tensor.expand_shape %extracted_slice_283 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_285 = tensor.extract_slice %arg0[3279163392] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_286 = tensor.expand_shape %extracted_slice_285 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_287 = tensor.extract_slice %arg0[3324252160] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_288 = tensor.expand_shape %extracted_slice_287 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_289 = tensor.extract_slice %arg0[3369340928] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_290 = tensor.expand_shape %extracted_slice_289 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_291 = tensor.extract_slice %arg0[3386118144] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_292 = tensor.expand_shape %extracted_slice_291 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_293 = tensor.extract_slice %arg0[3402895360] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_294 = tensor.expand_shape %extracted_slice_293 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_295 = tensor.extract_slice %arg0[3419672576] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_296 = tensor.expand_shape %extracted_slice_295 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_297 = tensor.extract_slice %arg0[3436449792] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_298 = tensor.expand_shape %extracted_slice_297 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_299 = tensor.extract_slice %arg0[3481538560] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_300 = tensor.expand_shape %extracted_slice_299 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_301 = tensor.extract_slice %arg0[3526627328] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_302 = tensor.expand_shape %extracted_slice_301 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_303 = tensor.extract_slice %arg0[3571716096] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_304 = tensor.expand_shape %extracted_slice_303 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_305 = tensor.extract_slice %arg0[3588493312] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_306 = tensor.expand_shape %extracted_slice_305 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_307 = tensor.extract_slice %arg0[3605270528] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_308 = tensor.expand_shape %extracted_slice_307 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_309 = tensor.extract_slice %arg0[3622047744] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_310 = tensor.expand_shape %extracted_slice_309 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_311 = tensor.extract_slice %arg0[3638824960] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_312 = tensor.expand_shape %extracted_slice_311 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_313 = tensor.extract_slice %arg0[3683913728] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_314 = tensor.expand_shape %extracted_slice_313 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_315 = tensor.extract_slice %arg0[3729002496] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_316 = tensor.expand_shape %extracted_slice_315 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_317 = tensor.extract_slice %arg0[3774091264] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_318 = tensor.expand_shape %extracted_slice_317 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_319 = tensor.extract_slice %arg0[3790868480] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_320 = tensor.expand_shape %extracted_slice_319 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_321 = tensor.extract_slice %arg0[3807645696] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_322 = tensor.expand_shape %extracted_slice_321 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_323 = tensor.extract_slice %arg0[3824422912] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_324 = tensor.expand_shape %extracted_slice_323 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_325 = tensor.extract_slice %arg0[3841200128] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_326 = tensor.expand_shape %extracted_slice_325 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_327 = tensor.extract_slice %arg0[3886288896] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_328 = tensor.expand_shape %extracted_slice_327 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_329 = tensor.extract_slice %arg0[3931377664] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_330 = tensor.expand_shape %extracted_slice_329 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_331 = tensor.extract_slice %arg0[3976466432] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_332 = tensor.expand_shape %extracted_slice_331 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_333 = tensor.extract_slice %arg0[3993243648] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_334 = tensor.expand_shape %extracted_slice_333 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_335 = tensor.extract_slice %arg0[4010020864] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_336 = tensor.expand_shape %extracted_slice_335 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_337 = tensor.extract_slice %arg0[4026798080] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_338 = tensor.expand_shape %extracted_slice_337 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_339 = tensor.extract_slice %arg0[4043575296] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_340 = tensor.expand_shape %extracted_slice_339 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_341 = tensor.extract_slice %arg0[4088664064] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_342 = tensor.expand_shape %extracted_slice_341 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_343 = tensor.extract_slice %arg0[4133752832] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_344 = tensor.expand_shape %extracted_slice_343 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_345 = tensor.extract_slice %arg0[4178841600] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_346 = tensor.expand_shape %extracted_slice_345 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_347 = tensor.extract_slice %arg0[4195618816] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_348 = tensor.expand_shape %extracted_slice_347 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_349 = tensor.extract_slice %arg0[4212396032] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_350 = tensor.expand_shape %extracted_slice_349 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_351 = tensor.extract_slice %arg0[4229173248] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_352 = tensor.expand_shape %extracted_slice_351 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_353 = tensor.extract_slice %arg0[4245950464] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_354 = tensor.expand_shape %extracted_slice_353 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_355 = tensor.extract_slice %arg0[4291039232] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_356 = tensor.expand_shape %extracted_slice_355 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_357 = tensor.extract_slice %arg0[4336128000] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_358 = tensor.expand_shape %extracted_slice_357 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_359 = tensor.extract_slice %arg0[4381216768] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_360 = tensor.expand_shape %extracted_slice_359 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_361 = tensor.extract_slice %arg0[4397993984] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_362 = tensor.expand_shape %extracted_slice_361 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_363 = tensor.extract_slice %arg0[4414771200] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_364 = tensor.expand_shape %extracted_slice_363 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_365 = tensor.extract_slice %arg0[4431548416] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_366 = tensor.expand_shape %extracted_slice_365 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_367 = tensor.extract_slice %arg0[4448325632] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_368 = tensor.expand_shape %extracted_slice_367 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_369 = tensor.extract_slice %arg0[4493414400] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_370 = tensor.expand_shape %extracted_slice_369 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_371 = tensor.extract_slice %arg0[4538503168] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_372 = tensor.expand_shape %extracted_slice_371 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_373 = tensor.extract_slice %arg0[4583591936] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_374 = tensor.expand_shape %extracted_slice_373 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_375 = tensor.extract_slice %arg0[4600369152] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_376 = tensor.expand_shape %extracted_slice_375 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_377 = tensor.extract_slice %arg0[4617146368] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_378 = tensor.expand_shape %extracted_slice_377 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_379 = tensor.extract_slice %arg0[4633923584] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_380 = tensor.expand_shape %extracted_slice_379 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_381 = tensor.extract_slice %arg0[4650700800] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_382 = tensor.expand_shape %extracted_slice_381 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_383 = tensor.extract_slice %arg0[4695789568] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_384 = tensor.expand_shape %extracted_slice_383 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_385 = tensor.extract_slice %arg0[4740878336] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_386 = tensor.expand_shape %extracted_slice_385 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_387 = tensor.extract_slice %arg0[4785967104] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_388 = tensor.expand_shape %extracted_slice_387 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_389 = tensor.extract_slice %arg0[4802744320] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_390 = tensor.expand_shape %extracted_slice_389 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_391 = tensor.extract_slice %arg0[4819521536] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_392 = tensor.expand_shape %extracted_slice_391 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_393 = tensor.extract_slice %arg0[4836298752] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_394 = tensor.expand_shape %extracted_slice_393 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_395 = tensor.extract_slice %arg0[4853075968] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_396 = tensor.expand_shape %extracted_slice_395 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_397 = tensor.extract_slice %arg0[4898164736] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_398 = tensor.expand_shape %extracted_slice_397 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_399 = tensor.extract_slice %arg0[4943253504] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_400 = tensor.expand_shape %extracted_slice_399 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_401 = tensor.extract_slice %arg0[4988342272] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_402 = tensor.expand_shape %extracted_slice_401 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_403 = tensor.extract_slice %arg0[5005119488] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_404 = tensor.expand_shape %extracted_slice_403 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_405 = tensor.extract_slice %arg0[5021896704] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_406 = tensor.expand_shape %extracted_slice_405 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_407 = tensor.extract_slice %arg0[5038673920] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_408 = tensor.expand_shape %extracted_slice_407 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_409 = tensor.extract_slice %arg0[5055451136] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_410 = tensor.expand_shape %extracted_slice_409 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_411 = tensor.extract_slice %arg0[5100539904] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_412 = tensor.expand_shape %extracted_slice_411 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_413 = tensor.extract_slice %arg0[5145628672] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_414 = tensor.expand_shape %extracted_slice_413 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_415 = tensor.extract_slice %arg0[5190717440] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_416 = tensor.expand_shape %extracted_slice_415 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_417 = tensor.extract_slice %arg0[5207494656] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_418 = tensor.expand_shape %extracted_slice_417 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_419 = tensor.extract_slice %arg0[5224271872] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_420 = tensor.expand_shape %extracted_slice_419 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_421 = tensor.extract_slice %arg0[5241049088] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_422 = tensor.expand_shape %extracted_slice_421 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_423 = tensor.extract_slice %arg0[5257826304] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_424 = tensor.expand_shape %extracted_slice_423 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_425 = tensor.extract_slice %arg0[5302915072] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_426 = tensor.expand_shape %extracted_slice_425 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_427 = tensor.extract_slice %arg0[5348003840] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_428 = tensor.expand_shape %extracted_slice_427 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_429 = tensor.extract_slice %arg0[5393092608] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_430 = tensor.expand_shape %extracted_slice_429 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_431 = tensor.extract_slice %arg0[5409869824] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_432 = tensor.expand_shape %extracted_slice_431 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_433 = tensor.extract_slice %arg0[5426647040] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_434 = tensor.expand_shape %extracted_slice_433 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_435 = tensor.extract_slice %arg0[5443424256] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_436 = tensor.expand_shape %extracted_slice_435 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_437 = tensor.extract_slice %arg0[5460201472] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_438 = tensor.expand_shape %extracted_slice_437 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_439 = tensor.extract_slice %arg0[5505290240] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_440 = tensor.expand_shape %extracted_slice_439 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_441 = tensor.extract_slice %arg0[5550379008] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_442 = tensor.expand_shape %extracted_slice_441 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_443 = tensor.extract_slice %arg0[5595467776] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_444 = tensor.expand_shape %extracted_slice_443 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_445 = tensor.extract_slice %arg0[5612244992] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_446 = tensor.expand_shape %extracted_slice_445 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_447 = tensor.extract_slice %arg0[5629022208] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_448 = tensor.expand_shape %extracted_slice_447 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_449 = tensor.extract_slice %arg0[5645799424] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_450 = tensor.expand_shape %extracted_slice_449 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_451 = tensor.extract_slice %arg0[5662576640] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_452 = tensor.expand_shape %extracted_slice_451 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_453 = tensor.extract_slice %arg0[5707665408] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_454 = tensor.expand_shape %extracted_slice_453 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_455 = tensor.extract_slice %arg0[5752754176] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_456 = tensor.expand_shape %extracted_slice_455 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_457 = tensor.extract_slice %arg0[5797842944] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_458 = tensor.expand_shape %extracted_slice_457 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_459 = tensor.extract_slice %arg0[5814620160] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_460 = tensor.expand_shape %extracted_slice_459 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_461 = tensor.extract_slice %arg0[5831397376] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_462 = tensor.expand_shape %extracted_slice_461 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_463 = tensor.extract_slice %arg0[5848174592] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_464 = tensor.expand_shape %extracted_slice_463 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_465 = tensor.extract_slice %arg0[5864951808] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_466 = tensor.expand_shape %extracted_slice_465 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_467 = tensor.extract_slice %arg0[5910040576] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_468 = tensor.expand_shape %extracted_slice_467 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_469 = tensor.extract_slice %arg0[5955129344] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_470 = tensor.expand_shape %extracted_slice_469 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_471 = tensor.extract_slice %arg0[6000218112] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_472 = tensor.expand_shape %extracted_slice_471 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_473 = tensor.extract_slice %arg0[6016995328] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_474 = tensor.expand_shape %extracted_slice_473 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_475 = tensor.extract_slice %arg0[6033772544] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_476 = tensor.expand_shape %extracted_slice_475 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_477 = tensor.extract_slice %arg0[6050549760] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_478 = tensor.expand_shape %extracted_slice_477 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_479 = tensor.extract_slice %arg0[6067326976] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_480 = tensor.expand_shape %extracted_slice_479 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_481 = tensor.extract_slice %arg0[6112415744] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_482 = tensor.expand_shape %extracted_slice_481 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_483 = tensor.extract_slice %arg0[6157504512] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_484 = tensor.expand_shape %extracted_slice_483 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_485 = tensor.extract_slice %arg0[6202593280] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_486 = tensor.expand_shape %extracted_slice_485 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_487 = tensor.extract_slice %arg0[6219370496] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_488 = tensor.expand_shape %extracted_slice_487 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_489 = tensor.extract_slice %arg0[6236147712] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_490 = tensor.expand_shape %extracted_slice_489 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_491 = tensor.extract_slice %arg0[6252924928] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_492 = tensor.expand_shape %extracted_slice_491 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_493 = tensor.extract_slice %arg0[6269702144] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_494 = tensor.expand_shape %extracted_slice_493 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_495 = tensor.extract_slice %arg0[6314790912] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_496 = tensor.expand_shape %extracted_slice_495 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_497 = tensor.extract_slice %arg0[6359879680] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_498 = tensor.expand_shape %extracted_slice_497 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_499 = tensor.extract_slice %arg0[6404968448] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_500 = tensor.expand_shape %extracted_slice_499 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_501 = tensor.extract_slice %arg0[6421745664] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_502 = tensor.expand_shape %extracted_slice_501 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_503 = tensor.extract_slice %arg0[6438522880] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_504 = tensor.expand_shape %extracted_slice_503 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_505 = tensor.extract_slice %arg0[6455300096] [16777216] [1] : tensor<6755192832xf32> to tensor<16777216xf32>
    %expanded_506 = tensor.expand_shape %extracted_slice_505 [[0, 1]] : tensor<16777216xf32> into tensor<4096x4096xf32>
    %extracted_slice_507 = tensor.extract_slice %arg0[6472077312] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_508 = tensor.expand_shape %extracted_slice_507 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_509 = tensor.extract_slice %arg0[6517166080] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_510 = tensor.expand_shape %extracted_slice_509 [[0, 1]] : tensor<45088768xf32> into tensor<11008x4096xf32>
    %extracted_slice_511 = tensor.extract_slice %arg0[6562254848] [45088768] [1] : tensor<6755192832xf32> to tensor<45088768xf32>
    %expanded_512 = tensor.expand_shape %extracted_slice_511 [[0, 1]] : tensor<45088768xf32> into tensor<4096x11008xf32>
    %extracted_slice_513 = tensor.extract_slice %arg0[6607343616] [131072000] [1] : tensor<6755192832xf32> to tensor<131072000xf32>
    %expanded_514 = tensor.expand_shape %extracted_slice_513 [[0, 1]] : tensor<131072000xf32> into tensor<32000x4096xf32>
    %extracted_slice_515 = tensor.extract_slice %arg0[6738415616] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_516 = tensor.expand_shape %extracted_slice_515 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_517 = tensor.extract_slice %arg0[6738677760] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_518 = tensor.expand_shape %extracted_slice_517 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_519 = tensor.extract_slice %arg0[6738939904] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_520 = tensor.expand_shape %extracted_slice_519 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_521 = tensor.extract_slice %arg0[6739202048] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_522 = tensor.expand_shape %extracted_slice_521 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_523 = tensor.extract_slice %arg0[6739464192] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_524 = tensor.expand_shape %extracted_slice_523 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_525 = tensor.extract_slice %arg0[6739726336] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_526 = tensor.expand_shape %extracted_slice_525 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_527 = tensor.extract_slice %arg0[6739988480] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_528 = tensor.expand_shape %extracted_slice_527 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_529 = tensor.extract_slice %arg0[6740250624] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_530 = tensor.expand_shape %extracted_slice_529 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_531 = tensor.extract_slice %arg0[6740512768] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_532 = tensor.expand_shape %extracted_slice_531 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_533 = tensor.extract_slice %arg0[6740774912] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_534 = tensor.expand_shape %extracted_slice_533 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_535 = tensor.extract_slice %arg0[6741037056] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_536 = tensor.expand_shape %extracted_slice_535 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_537 = tensor.extract_slice %arg0[6741299200] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_538 = tensor.expand_shape %extracted_slice_537 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_539 = tensor.extract_slice %arg0[6741561344] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_540 = tensor.expand_shape %extracted_slice_539 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_541 = tensor.extract_slice %arg0[6741823488] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_542 = tensor.expand_shape %extracted_slice_541 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_543 = tensor.extract_slice %arg0[6742085632] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_544 = tensor.expand_shape %extracted_slice_543 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_545 = tensor.extract_slice %arg0[6742347776] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_546 = tensor.expand_shape %extracted_slice_545 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_547 = tensor.extract_slice %arg0[6742609920] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_548 = tensor.expand_shape %extracted_slice_547 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_549 = tensor.extract_slice %arg0[6742872064] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_550 = tensor.expand_shape %extracted_slice_549 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_551 = tensor.extract_slice %arg0[6743134208] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_552 = tensor.expand_shape %extracted_slice_551 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_553 = tensor.extract_slice %arg0[6743396352] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_554 = tensor.expand_shape %extracted_slice_553 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_555 = tensor.extract_slice %arg0[6743658496] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_556 = tensor.expand_shape %extracted_slice_555 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_557 = tensor.extract_slice %arg0[6743920640] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_558 = tensor.expand_shape %extracted_slice_557 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_559 = tensor.extract_slice %arg0[6744182784] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_560 = tensor.expand_shape %extracted_slice_559 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_561 = tensor.extract_slice %arg0[6744444928] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_562 = tensor.expand_shape %extracted_slice_561 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_563 = tensor.extract_slice %arg0[6744707072] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_564 = tensor.expand_shape %extracted_slice_563 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_565 = tensor.extract_slice %arg0[6744969216] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_566 = tensor.expand_shape %extracted_slice_565 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_567 = tensor.extract_slice %arg0[6745231360] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_568 = tensor.expand_shape %extracted_slice_567 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_569 = tensor.extract_slice %arg0[6745493504] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_570 = tensor.expand_shape %extracted_slice_569 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_571 = tensor.extract_slice %arg0[6745755648] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_572 = tensor.expand_shape %extracted_slice_571 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_573 = tensor.extract_slice %arg0[6746017792] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_574 = tensor.expand_shape %extracted_slice_573 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_575 = tensor.extract_slice %arg0[6746279936] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_576 = tensor.expand_shape %extracted_slice_575 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_577 = tensor.extract_slice %arg0[6746542080] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_578 = tensor.expand_shape %extracted_slice_577 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_579 = tensor.extract_slice %arg0[6746804224] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_580 = tensor.expand_shape %extracted_slice_579 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_581 = tensor.extract_slice %arg0[6747066368] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_582 = tensor.expand_shape %extracted_slice_581 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_583 = tensor.extract_slice %arg0[6747328512] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_584 = tensor.expand_shape %extracted_slice_583 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_585 = tensor.extract_slice %arg0[6747590656] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_586 = tensor.expand_shape %extracted_slice_585 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_587 = tensor.extract_slice %arg0[6747852800] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_588 = tensor.expand_shape %extracted_slice_587 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_589 = tensor.extract_slice %arg0[6748114944] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_590 = tensor.expand_shape %extracted_slice_589 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_591 = tensor.extract_slice %arg0[6748377088] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_592 = tensor.expand_shape %extracted_slice_591 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_593 = tensor.extract_slice %arg0[6748639232] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_594 = tensor.expand_shape %extracted_slice_593 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_595 = tensor.extract_slice %arg0[6748901376] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_596 = tensor.expand_shape %extracted_slice_595 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_597 = tensor.extract_slice %arg0[6749163520] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_598 = tensor.expand_shape %extracted_slice_597 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_599 = tensor.extract_slice %arg0[6749425664] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_600 = tensor.expand_shape %extracted_slice_599 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_601 = tensor.extract_slice %arg0[6749687808] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_602 = tensor.expand_shape %extracted_slice_601 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_603 = tensor.extract_slice %arg0[6749949952] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_604 = tensor.expand_shape %extracted_slice_603 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_605 = tensor.extract_slice %arg0[6750212096] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_606 = tensor.expand_shape %extracted_slice_605 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_607 = tensor.extract_slice %arg0[6750474240] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_608 = tensor.expand_shape %extracted_slice_607 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_609 = tensor.extract_slice %arg0[6750736384] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_610 = tensor.expand_shape %extracted_slice_609 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_611 = tensor.extract_slice %arg0[6750998528] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_612 = tensor.expand_shape %extracted_slice_611 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_613 = tensor.extract_slice %arg0[6751260672] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_614 = tensor.expand_shape %extracted_slice_613 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_615 = tensor.extract_slice %arg0[6751522816] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_616 = tensor.expand_shape %extracted_slice_615 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_617 = tensor.extract_slice %arg0[6751784960] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_618 = tensor.expand_shape %extracted_slice_617 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_619 = tensor.extract_slice %arg0[6752047104] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_620 = tensor.expand_shape %extracted_slice_619 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_621 = tensor.extract_slice %arg0[6752309248] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_622 = tensor.expand_shape %extracted_slice_621 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_623 = tensor.extract_slice %arg0[6752571392] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_624 = tensor.expand_shape %extracted_slice_623 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_625 = tensor.extract_slice %arg0[6752833536] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_626 = tensor.expand_shape %extracted_slice_625 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_627 = tensor.extract_slice %arg0[6753095680] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_628 = tensor.expand_shape %extracted_slice_627 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_629 = tensor.extract_slice %arg0[6753357824] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_630 = tensor.expand_shape %extracted_slice_629 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_631 = tensor.extract_slice %arg0[6753619968] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_632 = tensor.expand_shape %extracted_slice_631 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_633 = tensor.extract_slice %arg0[6753882112] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_634 = tensor.expand_shape %extracted_slice_633 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_635 = tensor.extract_slice %arg0[6754144256] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_636 = tensor.expand_shape %extracted_slice_635 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_637 = tensor.extract_slice %arg0[6754406400] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_638 = tensor.expand_shape %extracted_slice_637 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_639 = tensor.extract_slice %arg0[6754668544] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_640 = tensor.expand_shape %extracted_slice_639 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %extracted_slice_641 = tensor.extract_slice %arg0[6754930688] [262144] [1] : tensor<6755192832xf32> to tensor<262144xf32>
    %expanded_642 = tensor.expand_shape %extracted_slice_641 [[0, 1, 2, 3]] : tensor<262144xf32> into tensor<1x1x2048x128xf32>
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi64>
    %0 = "tosa.reshape"(%cst) {new_shape = array<i64: 1, 80>} : (tensor<80xi64>) -> tensor<1x80xi64>
    %1 = "tosa.reshape"(%0) {new_shape = array<i64: 1, 80>} : (tensor<1x80xi64>) -> tensor<1x80xi64>
    %2 = "tosa.cast"(%arg1) : (tensor<1x80xi64>) -> tensor<1x80xi32>
    %3 = "tosa.reshape"(%expanded) {new_shape = array<i64: 1, 32000, 4096>} : (tensor<32000x4096xf32>) -> tensor<1x32000x4096xf32>
    %4 = "tosa.gather"(%3, %2) : (tensor<1x32000x4096xf32>, tensor<1x80xi32>) -> tensor<1x80x4096xf32>
    %5 = "tosa.reshape"(%4) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %cst_643 = arith.constant dense<true> : tensor<1x80xi1>
    %cst_644 = arith.constant dense<-3.40282347E+38> : tensor<80x80xf32>
    %cst_645 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi64>
    %6 = "tosa.const"() {value = dense<1> : tensor<80xi64>} : () -> tensor<80xi64>
    %7 = "tosa.add"(%cst_645, %6) : (tensor<80xi64>, tensor<80xi64>) -> tensor<80xi64>
    %8 = "tosa.reshape"(%7) {new_shape = array<i64: 80, 1>} : (tensor<80xi64>) -> tensor<80x1xi64>
    %9 = tensor.empty() : tensor<80x80xi1>
    %10 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cst_645, %8 : tensor<80xi64>, tensor<80x1xi64>) outs(%9 : tensor<80x80xi1>) {
    ^bb0(%in: i64, %in_1454: i64, %out: i1):
      %4172 = arith.cmpi slt, %in, %in_1454 : i64
      linalg.yield %4172 : i1
    } -> tensor<80x80xi1>
    %cst_646 = arith.constant 0.000000e+00 : f32
    %11 = tensor.empty() : tensor<80x80xf32>
    %12 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_644, %10 : tensor<80x80xf32>, tensor<80x80xi1>) outs(%11 : tensor<80x80xf32>) {
    ^bb0(%in: f32, %in_1454: i1, %out: f32):
      %4172 = arith.select %in_1454, %cst_646, %in : f32
      linalg.yield %4172 : f32
    } -> tensor<80x80xf32>
    %extracted_slice_647 = tensor.extract_slice %cst_643[0, 0] [1, 80] [1, 1] : tensor<1x80xi1> to tensor<1x80xi1>
    %13 = "tosa.reshape"(%extracted_slice_647) {new_shape = array<i64: 1, 1, 80>} : (tensor<1x80xi1>) -> tensor<1x1x80xi1>
    %14 = "tosa.reshape"(%13) {new_shape = array<i64: 1, 1, 1, 80>} : (tensor<1x1x80xi1>) -> tensor<1x1x1x80xi1>
    %extracted_slice_648 = tensor.extract_slice %14[0, 0, 0, 0] [1, 1, 1, 80] [1, 1, 1, 1] : tensor<1x1x1x80xi1> to tensor<1x1x1x80xi1>
    %15 = "tosa.const"() {value = dense<false> : tensor<1x1x80x80xi1>} : () -> tensor<1x1x80x80xi1>
    %16 = "tosa.add"(%extracted_slice_648, %15) : (tensor<1x1x1x80xi1>, tensor<1x1x80x80xi1>) -> tensor<1x1x80x80xi1>
    %17 = tensor.empty() : tensor<1x1x80x80xf32>
    %18 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16 : tensor<1x1x80x80xi1>) outs(%17 : tensor<1x1x80x80xf32>) {
    ^bb0(%in: i1, %out: f32):
      %4172 = arith.extui %in : i1 to i32
      %4173 = arith.sitofp %4172 : i32 to f32
      linalg.yield %4173 : f32
    } -> tensor<1x1x80x80xf32>
    %cst_649 = arith.constant 1.000000e+00 : f32
    %19 = tensor.empty() : tensor<1x1x80x80xf32>
    %20 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18 : tensor<1x1x80x80xf32>) outs(%19 : tensor<1x1x80x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.subf %cst_649, %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x1x80x80xf32>
    %21 = tensor.empty() : tensor<1x1x80x80xi1>
    %22 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20 : tensor<1x1x80x80xf32>) outs(%21 : tensor<1x1x80x80xi1>) {
    ^bb0(%in: f32, %out: i1):
      %4172 = arith.fptosi %in : f32 to i32
      %4173 = arith.trunci %4172 : i32 to i1
      linalg.yield %4173 : i1
    } -> tensor<1x1x80x80xi1>
    %cst_650 = arith.constant -3.40282347E+38 : f32
    %23 = tensor.empty() : tensor<1x1x80x80xf32>
    %24 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %22 : tensor<1x1x80x80xf32>, tensor<1x1x80x80xi1>) outs(%23 : tensor<1x1x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: i1, %out: f32):
      %4172 = arith.select %in_1454, %cst_650, %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x1x80x80xf32>
    %25 = "tosa.reshape"(%12) {new_shape = array<i64: 1, 80, 80>} : (tensor<80x80xf32>) -> tensor<1x80x80xf32>
    %26 = "tosa.reshape"(%25) {new_shape = array<i64: 1, 1, 80, 80>} : (tensor<1x80x80xf32>) -> tensor<1x1x80x80xf32>
    %extracted_slice_651 = tensor.extract_slice %26[0, 0, 0, 0] [1, 1, 80, 80] [1, 1, 1, 1] : tensor<1x1x80x80xf32> to tensor<1x1x80x80xf32>
    %extracted_slice_652 = tensor.extract_slice %extracted_slice_651[0, 0, 0, 0] [1, 1, 80, 80] [1, 1, 1, 1] : tensor<1x1x80x80xf32> to tensor<1x1x80x80xf32>
    %27 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1x80x80xf32>} : () -> tensor<1x1x80x80xf32>
    %28 = "tosa.add"(%extracted_slice_652, %27) : (tensor<1x1x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x1x80x80xf32>
    %29 = "tosa.add"(%24, %28) : (tensor<1x1x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x1x80x80xf32>
    %30 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32 = arith.constant 2 : i32
    %31 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<1x80x4096xf32>) outs(%30 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_653 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %32 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%31 : tensor<1x80x4096xf32>) outs(%cst_653 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %33 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %34 = "tosa.add"(%32, %33) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %35 = "tosa.rsqrt"(%34) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %36 = "tosa.mul"(%5, %35) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %37 = "tosa.reshape"(%extracted_slice) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %38 = "tosa.mul"(%37, %36) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %39 = tensor.empty() : tensor<4096x4096xf32>
    %40 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_66 : tensor<4096x4096xf32>) outs(%39 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %41 = "tosa.reshape"(%38) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_654 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %42 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%41, %40 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_654 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %43 = "tosa.reshape"(%42) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %44 = tensor.empty() : tensor<4096x4096xf32>
    %45 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_68 : tensor<4096x4096xf32>) outs(%44 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %46 = "tosa.reshape"(%38) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_655 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %47 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%46, %45 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_655 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %48 = "tosa.reshape"(%47) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %49 = tensor.empty() : tensor<4096x4096xf32>
    %50 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_70 : tensor<4096x4096xf32>) outs(%49 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %51 = "tosa.reshape"(%38) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_656 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %52 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%51, %50 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_656 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %53 = "tosa.reshape"(%52) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %54 = "tosa.reshape"(%43) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %55 = tensor.empty() : tensor<1x32x80x128xf32>
    %56 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%54 : tensor<1x80x32x128xf32>) outs(%55 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %57 = "tosa.reshape"(%48) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %58 = tensor.empty() : tensor<1x32x80x128xf32>
    %59 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57 : tensor<1x80x32x128xf32>) outs(%58 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %60 = "tosa.reshape"(%53) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %61 = tensor.empty() : tensor<1x32x80x128xf32>
    %62 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%60 : tensor<1x80x32x128xf32>) outs(%61 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_657 = tensor.extract_slice %expanded_516[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_658 = tensor.extract_slice %extracted_slice_657[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_659 = tensor.extract_slice %extracted_slice_658[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_660 = tensor.extract_slice %expanded_518[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_661 = tensor.extract_slice %extracted_slice_660[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_662 = tensor.extract_slice %extracted_slice_661[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %63 = tensor.empty() : tensor<1x80x128xf32>
    %64 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_659 : tensor<1x1x80x128xf32>) outs(%63 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %65 = tensor.empty() : tensor<80x128xf32>
    %66 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%64 : tensor<1x80x128xf32>) outs(%65 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %67 = tensor.empty() : tensor<1x80x128xf32>
    %68 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_662 : tensor<1x1x80x128xf32>) outs(%67 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %69 = tensor.empty() : tensor<80x128xf32>
    %70 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%68 : tensor<1x80x128xf32>) outs(%69 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %71 = tensor.empty() : tensor<1x80x128xf32>
    %72 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%71 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %66[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %73 = "tosa.reshape"(%72) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %74 = tensor.empty() : tensor<1x80x128xf32>
    %75 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%74 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %70[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %76 = "tosa.reshape"(%75) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %77 = "tosa.mul"(%56, %73) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_663 = tensor.extract_slice %56[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_664 = tensor.extract_slice %56[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %78 = tensor.empty() : tensor<1x32x80x64xf32>
    %79 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_664 : tensor<1x32x80x64xf32>) outs(%78 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %80 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice = tensor.insert_slice %79 into %80[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_665 = tensor.insert_slice %extracted_slice_663 into %inserted_slice[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %81 = "tosa.mul"(%inserted_slice_665, %76) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %82 = "tosa.add"(%77, %81) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %83 = "tosa.mul"(%59, %73) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_666 = tensor.extract_slice %59[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_667 = tensor.extract_slice %59[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %84 = tensor.empty() : tensor<1x32x80x64xf32>
    %85 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_667 : tensor<1x32x80x64xf32>) outs(%84 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %86 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_668 = tensor.insert_slice %85 into %86[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_669 = tensor.insert_slice %extracted_slice_666 into %inserted_slice_668[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %87 = "tosa.mul"(%inserted_slice_669, %76) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %88 = "tosa.add"(%83, %87) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %89 = tensor.empty() : tensor<1x32x128x80xf32>
    %90 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%88 : tensor<1x32x80x128xf32>) outs(%89 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %91 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %92 = "tosa.add"(%82, %91) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %93 = "tosa.reshape"(%92) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %94 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %95 = "tosa.add"(%90, %94) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %96 = "tosa.reshape"(%95) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %97 = "tosa.matmul"(%93, %96) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %98 = "tosa.reshape"(%97) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %99 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %100 = "tosa.reciprocal"(%99) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %101 = "tosa.mul"(%98, %100) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %102 = "tosa.add"(%101, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %103 = tensor.empty() : tensor<1x32x80x1xf32>
    %104 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%103 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %105 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%102 : tensor<1x32x80x80xf32>) outs(%103 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %106 = tensor.empty() : tensor<1x32x80x80xf32>
    %107 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %105 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%106 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %108 = tensor.empty() : tensor<1x32x80x1xf32>
    %109 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%108 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %110 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%107 : tensor<1x32x80x80xf32>) outs(%109 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %111 = tensor.empty() : tensor<1x32x80x80xf32>
    %112 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %110 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%111 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %113 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %114 = "tosa.add"(%112, %113) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %115 = "tosa.reshape"(%114) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %116 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %117 = "tosa.add"(%62, %116) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %118 = "tosa.reshape"(%117) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %119 = "tosa.matmul"(%115, %118) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %120 = "tosa.reshape"(%119) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %121 = tensor.empty() : tensor<1x80x32x128xf32>
    %122 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120 : tensor<1x32x80x128xf32>) outs(%121 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %123 = "tosa.identity"(%122) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %124 = "tosa.reshape"(%123) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %125 = tensor.empty() : tensor<4096x4096xf32>
    %126 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_72 : tensor<4096x4096xf32>) outs(%125 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %127 = "tosa.reshape"(%124) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_670 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %128 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%127, %126 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_670 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %129 = "tosa.reshape"(%128) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %130 = "tosa.add"(%5, %129) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %131 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_671 = arith.constant 2 : i32
    %132 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%130 : tensor<1x80x4096xf32>) outs(%131 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_671 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_672 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %133 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%132 : tensor<1x80x4096xf32>) outs(%cst_672 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %134 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %135 = "tosa.add"(%133, %134) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %136 = "tosa.rsqrt"(%135) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %137 = "tosa.mul"(%130, %136) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %138 = "tosa.reshape"(%extracted_slice_0) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %139 = "tosa.mul"(%138, %137) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %140 = tensor.empty() : tensor<4096x11008xf32>
    %141 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_74 : tensor<11008x4096xf32>) outs(%140 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %142 = "tosa.reshape"(%139) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_673 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %143 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%142, %141 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_673 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %144 = "tosa.reshape"(%143) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %145 = tensor.empty() : tensor<1x80x11008xf32>
    %146 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%144 : tensor<1x80x11008xf32>) outs(%145 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %147 = tensor.empty() : tensor<4096x11008xf32>
    %148 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_76 : tensor<11008x4096xf32>) outs(%147 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %149 = "tosa.reshape"(%139) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_674 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %150 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%149, %148 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_674 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %151 = "tosa.reshape"(%150) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %152 = "tosa.mul"(%146, %151) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %153 = tensor.empty() : tensor<11008x4096xf32>
    %154 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_78 : tensor<4096x11008xf32>) outs(%153 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %155 = "tosa.reshape"(%152) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_675 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %156 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%155, %154 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_675 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %157 = "tosa.reshape"(%156) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %158 = "tosa.add"(%130, %157) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %159 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_676 = arith.constant 2 : i32
    %160 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%158 : tensor<1x80x4096xf32>) outs(%159 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_676 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_677 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %161 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%160 : tensor<1x80x4096xf32>) outs(%cst_677 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %162 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %163 = "tosa.add"(%161, %162) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %164 = "tosa.rsqrt"(%163) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %165 = "tosa.mul"(%158, %164) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %166 = "tosa.reshape"(%extracted_slice_1) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %167 = "tosa.mul"(%166, %165) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %168 = tensor.empty() : tensor<4096x4096xf32>
    %169 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_80 : tensor<4096x4096xf32>) outs(%168 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %170 = "tosa.reshape"(%167) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_678 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %171 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%170, %169 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_678 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %172 = "tosa.reshape"(%171) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %173 = tensor.empty() : tensor<4096x4096xf32>
    %174 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_82 : tensor<4096x4096xf32>) outs(%173 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %175 = "tosa.reshape"(%167) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_679 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %176 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%175, %174 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_679 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %177 = "tosa.reshape"(%176) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %178 = tensor.empty() : tensor<4096x4096xf32>
    %179 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_84 : tensor<4096x4096xf32>) outs(%178 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %180 = "tosa.reshape"(%167) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_680 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %181 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%180, %179 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_680 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %182 = "tosa.reshape"(%181) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %183 = "tosa.reshape"(%172) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %184 = tensor.empty() : tensor<1x32x80x128xf32>
    %185 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%183 : tensor<1x80x32x128xf32>) outs(%184 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %186 = "tosa.reshape"(%177) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %187 = tensor.empty() : tensor<1x32x80x128xf32>
    %188 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%186 : tensor<1x80x32x128xf32>) outs(%187 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %189 = "tosa.reshape"(%182) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %190 = tensor.empty() : tensor<1x32x80x128xf32>
    %191 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%189 : tensor<1x80x32x128xf32>) outs(%190 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_681 = tensor.extract_slice %expanded_520[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_682 = tensor.extract_slice %extracted_slice_681[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_683 = tensor.extract_slice %extracted_slice_682[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_684 = tensor.extract_slice %expanded_522[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_685 = tensor.extract_slice %extracted_slice_684[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_686 = tensor.extract_slice %extracted_slice_685[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %192 = tensor.empty() : tensor<1x80x128xf32>
    %193 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_683 : tensor<1x1x80x128xf32>) outs(%192 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %194 = tensor.empty() : tensor<80x128xf32>
    %195 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%193 : tensor<1x80x128xf32>) outs(%194 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %196 = tensor.empty() : tensor<1x80x128xf32>
    %197 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_686 : tensor<1x1x80x128xf32>) outs(%196 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %198 = tensor.empty() : tensor<80x128xf32>
    %199 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%197 : tensor<1x80x128xf32>) outs(%198 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %200 = tensor.empty() : tensor<1x80x128xf32>
    %201 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%200 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %195[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %202 = "tosa.reshape"(%201) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %203 = tensor.empty() : tensor<1x80x128xf32>
    %204 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%203 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %199[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %205 = "tosa.reshape"(%204) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %206 = "tosa.mul"(%185, %202) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_687 = tensor.extract_slice %185[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_688 = tensor.extract_slice %185[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %207 = tensor.empty() : tensor<1x32x80x64xf32>
    %208 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_688 : tensor<1x32x80x64xf32>) outs(%207 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %209 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_689 = tensor.insert_slice %208 into %209[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_690 = tensor.insert_slice %extracted_slice_687 into %inserted_slice_689[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %210 = "tosa.mul"(%inserted_slice_690, %205) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %211 = "tosa.add"(%206, %210) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %212 = "tosa.mul"(%188, %202) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_691 = tensor.extract_slice %188[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_692 = tensor.extract_slice %188[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %213 = tensor.empty() : tensor<1x32x80x64xf32>
    %214 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_692 : tensor<1x32x80x64xf32>) outs(%213 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %215 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_693 = tensor.insert_slice %214 into %215[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_694 = tensor.insert_slice %extracted_slice_691 into %inserted_slice_693[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %216 = "tosa.mul"(%inserted_slice_694, %205) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %217 = "tosa.add"(%212, %216) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %218 = tensor.empty() : tensor<1x32x128x80xf32>
    %219 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%217 : tensor<1x32x80x128xf32>) outs(%218 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %220 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %221 = "tosa.add"(%211, %220) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %222 = "tosa.reshape"(%221) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %223 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %224 = "tosa.add"(%219, %223) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %225 = "tosa.reshape"(%224) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %226 = "tosa.matmul"(%222, %225) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %227 = "tosa.reshape"(%226) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %228 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %229 = "tosa.reciprocal"(%228) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %230 = "tosa.mul"(%227, %229) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %231 = "tosa.add"(%230, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %232 = tensor.empty() : tensor<1x32x80x1xf32>
    %233 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%232 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %234 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%231 : tensor<1x32x80x80xf32>) outs(%232 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %235 = tensor.empty() : tensor<1x32x80x80xf32>
    %236 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%231, %234 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%235 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %237 = tensor.empty() : tensor<1x32x80x1xf32>
    %238 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%237 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %239 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%236 : tensor<1x32x80x80xf32>) outs(%238 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %240 = tensor.empty() : tensor<1x32x80x80xf32>
    %241 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%236, %239 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%240 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %242 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %243 = "tosa.add"(%241, %242) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %244 = "tosa.reshape"(%243) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %245 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %246 = "tosa.add"(%191, %245) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %247 = "tosa.reshape"(%246) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %248 = "tosa.matmul"(%244, %247) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %249 = "tosa.reshape"(%248) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %250 = tensor.empty() : tensor<1x80x32x128xf32>
    %251 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249 : tensor<1x32x80x128xf32>) outs(%250 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %252 = "tosa.identity"(%251) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %253 = "tosa.reshape"(%252) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %254 = tensor.empty() : tensor<4096x4096xf32>
    %255 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_86 : tensor<4096x4096xf32>) outs(%254 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %256 = "tosa.reshape"(%253) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_695 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %257 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%256, %255 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_695 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %258 = "tosa.reshape"(%257) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %259 = "tosa.add"(%158, %258) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %260 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_696 = arith.constant 2 : i32
    %261 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%259 : tensor<1x80x4096xf32>) outs(%260 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_696 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_697 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %262 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%261 : tensor<1x80x4096xf32>) outs(%cst_697 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %263 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %264 = "tosa.add"(%262, %263) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %265 = "tosa.rsqrt"(%264) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %266 = "tosa.mul"(%259, %265) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %267 = "tosa.reshape"(%extracted_slice_2) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %268 = "tosa.mul"(%267, %266) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %269 = tensor.empty() : tensor<4096x11008xf32>
    %270 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_88 : tensor<11008x4096xf32>) outs(%269 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %271 = "tosa.reshape"(%268) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_698 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %272 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%271, %270 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_698 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %273 = "tosa.reshape"(%272) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %274 = tensor.empty() : tensor<1x80x11008xf32>
    %275 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%273 : tensor<1x80x11008xf32>) outs(%274 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %276 = tensor.empty() : tensor<4096x11008xf32>
    %277 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_90 : tensor<11008x4096xf32>) outs(%276 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %278 = "tosa.reshape"(%268) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_699 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %279 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%278, %277 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_699 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %280 = "tosa.reshape"(%279) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %281 = "tosa.mul"(%275, %280) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %282 = tensor.empty() : tensor<11008x4096xf32>
    %283 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_92 : tensor<4096x11008xf32>) outs(%282 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %284 = "tosa.reshape"(%281) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_700 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %285 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%284, %283 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_700 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %286 = "tosa.reshape"(%285) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %287 = "tosa.add"(%259, %286) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %288 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_701 = arith.constant 2 : i32
    %289 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%287 : tensor<1x80x4096xf32>) outs(%288 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_701 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_702 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %290 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%289 : tensor<1x80x4096xf32>) outs(%cst_702 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %291 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %292 = "tosa.add"(%290, %291) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %293 = "tosa.rsqrt"(%292) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %294 = "tosa.mul"(%287, %293) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %295 = "tosa.reshape"(%extracted_slice_3) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %296 = "tosa.mul"(%295, %294) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %297 = tensor.empty() : tensor<4096x4096xf32>
    %298 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_94 : tensor<4096x4096xf32>) outs(%297 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %299 = "tosa.reshape"(%296) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_703 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %300 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%299, %298 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_703 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %301 = "tosa.reshape"(%300) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %302 = tensor.empty() : tensor<4096x4096xf32>
    %303 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_96 : tensor<4096x4096xf32>) outs(%302 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %304 = "tosa.reshape"(%296) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_704 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %305 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%304, %303 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_704 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %306 = "tosa.reshape"(%305) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %307 = tensor.empty() : tensor<4096x4096xf32>
    %308 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_98 : tensor<4096x4096xf32>) outs(%307 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %309 = "tosa.reshape"(%296) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_705 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %310 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%309, %308 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_705 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %311 = "tosa.reshape"(%310) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %312 = "tosa.reshape"(%301) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %313 = tensor.empty() : tensor<1x32x80x128xf32>
    %314 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%312 : tensor<1x80x32x128xf32>) outs(%313 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %315 = "tosa.reshape"(%306) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %316 = tensor.empty() : tensor<1x32x80x128xf32>
    %317 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%315 : tensor<1x80x32x128xf32>) outs(%316 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %318 = "tosa.reshape"(%311) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %319 = tensor.empty() : tensor<1x32x80x128xf32>
    %320 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%318 : tensor<1x80x32x128xf32>) outs(%319 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_706 = tensor.extract_slice %expanded_524[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_707 = tensor.extract_slice %extracted_slice_706[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_708 = tensor.extract_slice %extracted_slice_707[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_709 = tensor.extract_slice %expanded_526[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_710 = tensor.extract_slice %extracted_slice_709[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_711 = tensor.extract_slice %extracted_slice_710[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %321 = tensor.empty() : tensor<1x80x128xf32>
    %322 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_708 : tensor<1x1x80x128xf32>) outs(%321 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %323 = tensor.empty() : tensor<80x128xf32>
    %324 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%322 : tensor<1x80x128xf32>) outs(%323 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %325 = tensor.empty() : tensor<1x80x128xf32>
    %326 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_711 : tensor<1x1x80x128xf32>) outs(%325 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %327 = tensor.empty() : tensor<80x128xf32>
    %328 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%326 : tensor<1x80x128xf32>) outs(%327 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %329 = tensor.empty() : tensor<1x80x128xf32>
    %330 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%329 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %324[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %331 = "tosa.reshape"(%330) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %332 = tensor.empty() : tensor<1x80x128xf32>
    %333 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%332 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %328[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %334 = "tosa.reshape"(%333) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %335 = "tosa.mul"(%314, %331) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_712 = tensor.extract_slice %314[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_713 = tensor.extract_slice %314[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %336 = tensor.empty() : tensor<1x32x80x64xf32>
    %337 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_713 : tensor<1x32x80x64xf32>) outs(%336 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %338 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_714 = tensor.insert_slice %337 into %338[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_715 = tensor.insert_slice %extracted_slice_712 into %inserted_slice_714[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %339 = "tosa.mul"(%inserted_slice_715, %334) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %340 = "tosa.add"(%335, %339) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %341 = "tosa.mul"(%317, %331) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_716 = tensor.extract_slice %317[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_717 = tensor.extract_slice %317[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %342 = tensor.empty() : tensor<1x32x80x64xf32>
    %343 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_717 : tensor<1x32x80x64xf32>) outs(%342 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %344 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_718 = tensor.insert_slice %343 into %344[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_719 = tensor.insert_slice %extracted_slice_716 into %inserted_slice_718[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %345 = "tosa.mul"(%inserted_slice_719, %334) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %346 = "tosa.add"(%341, %345) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %347 = tensor.empty() : tensor<1x32x128x80xf32>
    %348 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%346 : tensor<1x32x80x128xf32>) outs(%347 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %349 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %350 = "tosa.add"(%340, %349) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %351 = "tosa.reshape"(%350) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %352 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %353 = "tosa.add"(%348, %352) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %354 = "tosa.reshape"(%353) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %355 = "tosa.matmul"(%351, %354) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %356 = "tosa.reshape"(%355) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %357 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %358 = "tosa.reciprocal"(%357) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %359 = "tosa.mul"(%356, %358) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %360 = "tosa.add"(%359, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %361 = tensor.empty() : tensor<1x32x80x1xf32>
    %362 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%361 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %363 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%360 : tensor<1x32x80x80xf32>) outs(%361 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %364 = tensor.empty() : tensor<1x32x80x80xf32>
    %365 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%360, %363 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%364 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %366 = tensor.empty() : tensor<1x32x80x1xf32>
    %367 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%366 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %368 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%365 : tensor<1x32x80x80xf32>) outs(%367 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %369 = tensor.empty() : tensor<1x32x80x80xf32>
    %370 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%365, %368 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%369 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %371 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %372 = "tosa.add"(%370, %371) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %373 = "tosa.reshape"(%372) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %374 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %375 = "tosa.add"(%320, %374) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %376 = "tosa.reshape"(%375) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %377 = "tosa.matmul"(%373, %376) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %378 = "tosa.reshape"(%377) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %379 = tensor.empty() : tensor<1x80x32x128xf32>
    %380 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%378 : tensor<1x32x80x128xf32>) outs(%379 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %381 = "tosa.identity"(%380) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %382 = "tosa.reshape"(%381) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %383 = tensor.empty() : tensor<4096x4096xf32>
    %384 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_100 : tensor<4096x4096xf32>) outs(%383 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %385 = "tosa.reshape"(%382) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_720 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %386 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%385, %384 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_720 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %387 = "tosa.reshape"(%386) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %388 = "tosa.add"(%287, %387) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %389 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_721 = arith.constant 2 : i32
    %390 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%388 : tensor<1x80x4096xf32>) outs(%389 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_721 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_722 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %391 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%390 : tensor<1x80x4096xf32>) outs(%cst_722 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %392 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %393 = "tosa.add"(%391, %392) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %394 = "tosa.rsqrt"(%393) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %395 = "tosa.mul"(%388, %394) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %396 = "tosa.reshape"(%extracted_slice_4) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %397 = "tosa.mul"(%396, %395) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %398 = tensor.empty() : tensor<4096x11008xf32>
    %399 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_102 : tensor<11008x4096xf32>) outs(%398 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %400 = "tosa.reshape"(%397) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_723 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %401 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%400, %399 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_723 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %402 = "tosa.reshape"(%401) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %403 = tensor.empty() : tensor<1x80x11008xf32>
    %404 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%402 : tensor<1x80x11008xf32>) outs(%403 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %405 = tensor.empty() : tensor<4096x11008xf32>
    %406 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_104 : tensor<11008x4096xf32>) outs(%405 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %407 = "tosa.reshape"(%397) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_724 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %408 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%407, %406 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_724 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %409 = "tosa.reshape"(%408) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %410 = "tosa.mul"(%404, %409) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %411 = tensor.empty() : tensor<11008x4096xf32>
    %412 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_106 : tensor<4096x11008xf32>) outs(%411 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %413 = "tosa.reshape"(%410) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_725 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %414 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%413, %412 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_725 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %415 = "tosa.reshape"(%414) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %416 = "tosa.add"(%388, %415) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %417 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_726 = arith.constant 2 : i32
    %418 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%416 : tensor<1x80x4096xf32>) outs(%417 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_726 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_727 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %419 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%418 : tensor<1x80x4096xf32>) outs(%cst_727 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %420 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %421 = "tosa.add"(%419, %420) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %422 = "tosa.rsqrt"(%421) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %423 = "tosa.mul"(%416, %422) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %424 = "tosa.reshape"(%extracted_slice_5) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %425 = "tosa.mul"(%424, %423) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %426 = tensor.empty() : tensor<4096x4096xf32>
    %427 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_108 : tensor<4096x4096xf32>) outs(%426 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %428 = "tosa.reshape"(%425) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_728 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %429 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%428, %427 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_728 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %430 = "tosa.reshape"(%429) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %431 = tensor.empty() : tensor<4096x4096xf32>
    %432 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_110 : tensor<4096x4096xf32>) outs(%431 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %433 = "tosa.reshape"(%425) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_729 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %434 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%433, %432 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_729 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %435 = "tosa.reshape"(%434) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %436 = tensor.empty() : tensor<4096x4096xf32>
    %437 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_112 : tensor<4096x4096xf32>) outs(%436 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %438 = "tosa.reshape"(%425) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_730 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %439 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%438, %437 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_730 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %440 = "tosa.reshape"(%439) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %441 = "tosa.reshape"(%430) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %442 = tensor.empty() : tensor<1x32x80x128xf32>
    %443 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%441 : tensor<1x80x32x128xf32>) outs(%442 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %444 = "tosa.reshape"(%435) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %445 = tensor.empty() : tensor<1x32x80x128xf32>
    %446 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%444 : tensor<1x80x32x128xf32>) outs(%445 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %447 = "tosa.reshape"(%440) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %448 = tensor.empty() : tensor<1x32x80x128xf32>
    %449 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%447 : tensor<1x80x32x128xf32>) outs(%448 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_731 = tensor.extract_slice %expanded_528[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_732 = tensor.extract_slice %extracted_slice_731[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_733 = tensor.extract_slice %extracted_slice_732[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_734 = tensor.extract_slice %expanded_530[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_735 = tensor.extract_slice %extracted_slice_734[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_736 = tensor.extract_slice %extracted_slice_735[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %450 = tensor.empty() : tensor<1x80x128xf32>
    %451 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_733 : tensor<1x1x80x128xf32>) outs(%450 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %452 = tensor.empty() : tensor<80x128xf32>
    %453 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%451 : tensor<1x80x128xf32>) outs(%452 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %454 = tensor.empty() : tensor<1x80x128xf32>
    %455 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_736 : tensor<1x1x80x128xf32>) outs(%454 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %456 = tensor.empty() : tensor<80x128xf32>
    %457 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%455 : tensor<1x80x128xf32>) outs(%456 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %458 = tensor.empty() : tensor<1x80x128xf32>
    %459 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%458 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %453[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %460 = "tosa.reshape"(%459) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %461 = tensor.empty() : tensor<1x80x128xf32>
    %462 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%461 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %457[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %463 = "tosa.reshape"(%462) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %464 = "tosa.mul"(%443, %460) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_737 = tensor.extract_slice %443[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_738 = tensor.extract_slice %443[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %465 = tensor.empty() : tensor<1x32x80x64xf32>
    %466 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_738 : tensor<1x32x80x64xf32>) outs(%465 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %467 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_739 = tensor.insert_slice %466 into %467[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_740 = tensor.insert_slice %extracted_slice_737 into %inserted_slice_739[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %468 = "tosa.mul"(%inserted_slice_740, %463) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %469 = "tosa.add"(%464, %468) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %470 = "tosa.mul"(%446, %460) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_741 = tensor.extract_slice %446[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_742 = tensor.extract_slice %446[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %471 = tensor.empty() : tensor<1x32x80x64xf32>
    %472 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_742 : tensor<1x32x80x64xf32>) outs(%471 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %473 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_743 = tensor.insert_slice %472 into %473[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_744 = tensor.insert_slice %extracted_slice_741 into %inserted_slice_743[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %474 = "tosa.mul"(%inserted_slice_744, %463) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %475 = "tosa.add"(%470, %474) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %476 = tensor.empty() : tensor<1x32x128x80xf32>
    %477 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%475 : tensor<1x32x80x128xf32>) outs(%476 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %478 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %479 = "tosa.add"(%469, %478) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %480 = "tosa.reshape"(%479) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %481 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %482 = "tosa.add"(%477, %481) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %483 = "tosa.reshape"(%482) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %484 = "tosa.matmul"(%480, %483) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %485 = "tosa.reshape"(%484) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %486 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %487 = "tosa.reciprocal"(%486) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %488 = "tosa.mul"(%485, %487) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %489 = "tosa.add"(%488, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %490 = tensor.empty() : tensor<1x32x80x1xf32>
    %491 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%490 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %492 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%489 : tensor<1x32x80x80xf32>) outs(%490 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %493 = tensor.empty() : tensor<1x32x80x80xf32>
    %494 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%489, %492 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%493 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %495 = tensor.empty() : tensor<1x32x80x1xf32>
    %496 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%495 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %497 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%494 : tensor<1x32x80x80xf32>) outs(%496 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %498 = tensor.empty() : tensor<1x32x80x80xf32>
    %499 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%494, %497 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%498 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %500 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %501 = "tosa.add"(%499, %500) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %502 = "tosa.reshape"(%501) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %503 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %504 = "tosa.add"(%449, %503) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %505 = "tosa.reshape"(%504) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %506 = "tosa.matmul"(%502, %505) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %507 = "tosa.reshape"(%506) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %508 = tensor.empty() : tensor<1x80x32x128xf32>
    %509 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%507 : tensor<1x32x80x128xf32>) outs(%508 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %510 = "tosa.identity"(%509) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %511 = "tosa.reshape"(%510) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %512 = tensor.empty() : tensor<4096x4096xf32>
    %513 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_114 : tensor<4096x4096xf32>) outs(%512 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %514 = "tosa.reshape"(%511) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_745 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %515 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%514, %513 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_745 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %516 = "tosa.reshape"(%515) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %517 = "tosa.add"(%416, %516) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %518 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_746 = arith.constant 2 : i32
    %519 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%517 : tensor<1x80x4096xf32>) outs(%518 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_746 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_747 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %520 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%519 : tensor<1x80x4096xf32>) outs(%cst_747 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %521 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %522 = "tosa.add"(%520, %521) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %523 = "tosa.rsqrt"(%522) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %524 = "tosa.mul"(%517, %523) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %525 = "tosa.reshape"(%extracted_slice_6) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %526 = "tosa.mul"(%525, %524) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %527 = tensor.empty() : tensor<4096x11008xf32>
    %528 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_116 : tensor<11008x4096xf32>) outs(%527 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %529 = "tosa.reshape"(%526) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_748 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %530 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%529, %528 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_748 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %531 = "tosa.reshape"(%530) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %532 = tensor.empty() : tensor<1x80x11008xf32>
    %533 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%531 : tensor<1x80x11008xf32>) outs(%532 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %534 = tensor.empty() : tensor<4096x11008xf32>
    %535 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_118 : tensor<11008x4096xf32>) outs(%534 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %536 = "tosa.reshape"(%526) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_749 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %537 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%536, %535 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_749 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %538 = "tosa.reshape"(%537) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %539 = "tosa.mul"(%533, %538) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %540 = tensor.empty() : tensor<11008x4096xf32>
    %541 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_120 : tensor<4096x11008xf32>) outs(%540 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %542 = "tosa.reshape"(%539) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_750 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %543 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%542, %541 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_750 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %544 = "tosa.reshape"(%543) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %545 = "tosa.add"(%517, %544) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %546 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_751 = arith.constant 2 : i32
    %547 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%545 : tensor<1x80x4096xf32>) outs(%546 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_751 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_752 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %548 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%547 : tensor<1x80x4096xf32>) outs(%cst_752 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %549 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %550 = "tosa.add"(%548, %549) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %551 = "tosa.rsqrt"(%550) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %552 = "tosa.mul"(%545, %551) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %553 = "tosa.reshape"(%extracted_slice_7) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %554 = "tosa.mul"(%553, %552) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %555 = tensor.empty() : tensor<4096x4096xf32>
    %556 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_122 : tensor<4096x4096xf32>) outs(%555 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %557 = "tosa.reshape"(%554) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_753 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %558 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%557, %556 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_753 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %559 = "tosa.reshape"(%558) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %560 = tensor.empty() : tensor<4096x4096xf32>
    %561 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_124 : tensor<4096x4096xf32>) outs(%560 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %562 = "tosa.reshape"(%554) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_754 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %563 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%562, %561 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_754 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %564 = "tosa.reshape"(%563) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %565 = tensor.empty() : tensor<4096x4096xf32>
    %566 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_126 : tensor<4096x4096xf32>) outs(%565 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %567 = "tosa.reshape"(%554) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_755 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %568 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%567, %566 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_755 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %569 = "tosa.reshape"(%568) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %570 = "tosa.reshape"(%559) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %571 = tensor.empty() : tensor<1x32x80x128xf32>
    %572 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%570 : tensor<1x80x32x128xf32>) outs(%571 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %573 = "tosa.reshape"(%564) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %574 = tensor.empty() : tensor<1x32x80x128xf32>
    %575 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%573 : tensor<1x80x32x128xf32>) outs(%574 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %576 = "tosa.reshape"(%569) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %577 = tensor.empty() : tensor<1x32x80x128xf32>
    %578 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%576 : tensor<1x80x32x128xf32>) outs(%577 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_756 = tensor.extract_slice %expanded_532[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_757 = tensor.extract_slice %extracted_slice_756[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_758 = tensor.extract_slice %extracted_slice_757[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_759 = tensor.extract_slice %expanded_534[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_760 = tensor.extract_slice %extracted_slice_759[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_761 = tensor.extract_slice %extracted_slice_760[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %579 = tensor.empty() : tensor<1x80x128xf32>
    %580 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_758 : tensor<1x1x80x128xf32>) outs(%579 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %581 = tensor.empty() : tensor<80x128xf32>
    %582 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%580 : tensor<1x80x128xf32>) outs(%581 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %583 = tensor.empty() : tensor<1x80x128xf32>
    %584 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_761 : tensor<1x1x80x128xf32>) outs(%583 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %585 = tensor.empty() : tensor<80x128xf32>
    %586 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%584 : tensor<1x80x128xf32>) outs(%585 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %587 = tensor.empty() : tensor<1x80x128xf32>
    %588 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%587 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %582[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %589 = "tosa.reshape"(%588) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %590 = tensor.empty() : tensor<1x80x128xf32>
    %591 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%590 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %586[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %592 = "tosa.reshape"(%591) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %593 = "tosa.mul"(%572, %589) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_762 = tensor.extract_slice %572[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_763 = tensor.extract_slice %572[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %594 = tensor.empty() : tensor<1x32x80x64xf32>
    %595 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_763 : tensor<1x32x80x64xf32>) outs(%594 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %596 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_764 = tensor.insert_slice %595 into %596[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_765 = tensor.insert_slice %extracted_slice_762 into %inserted_slice_764[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %597 = "tosa.mul"(%inserted_slice_765, %592) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %598 = "tosa.add"(%593, %597) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %599 = "tosa.mul"(%575, %589) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_766 = tensor.extract_slice %575[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_767 = tensor.extract_slice %575[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %600 = tensor.empty() : tensor<1x32x80x64xf32>
    %601 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_767 : tensor<1x32x80x64xf32>) outs(%600 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %602 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_768 = tensor.insert_slice %601 into %602[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_769 = tensor.insert_slice %extracted_slice_766 into %inserted_slice_768[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %603 = "tosa.mul"(%inserted_slice_769, %592) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %604 = "tosa.add"(%599, %603) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %605 = tensor.empty() : tensor<1x32x128x80xf32>
    %606 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%604 : tensor<1x32x80x128xf32>) outs(%605 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %607 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %608 = "tosa.add"(%598, %607) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %609 = "tosa.reshape"(%608) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %610 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %611 = "tosa.add"(%606, %610) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %612 = "tosa.reshape"(%611) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %613 = "tosa.matmul"(%609, %612) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %614 = "tosa.reshape"(%613) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %615 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %616 = "tosa.reciprocal"(%615) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %617 = "tosa.mul"(%614, %616) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %618 = "tosa.add"(%617, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %619 = tensor.empty() : tensor<1x32x80x1xf32>
    %620 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%619 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %621 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%618 : tensor<1x32x80x80xf32>) outs(%619 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %622 = tensor.empty() : tensor<1x32x80x80xf32>
    %623 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%618, %621 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%622 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %624 = tensor.empty() : tensor<1x32x80x1xf32>
    %625 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%624 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %626 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%623 : tensor<1x32x80x80xf32>) outs(%625 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %627 = tensor.empty() : tensor<1x32x80x80xf32>
    %628 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%623, %626 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%627 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %629 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %630 = "tosa.add"(%628, %629) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %631 = "tosa.reshape"(%630) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %632 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %633 = "tosa.add"(%578, %632) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %634 = "tosa.reshape"(%633) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %635 = "tosa.matmul"(%631, %634) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %636 = "tosa.reshape"(%635) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %637 = tensor.empty() : tensor<1x80x32x128xf32>
    %638 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%636 : tensor<1x32x80x128xf32>) outs(%637 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %639 = "tosa.identity"(%638) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %640 = "tosa.reshape"(%639) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %641 = tensor.empty() : tensor<4096x4096xf32>
    %642 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_128 : tensor<4096x4096xf32>) outs(%641 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %643 = "tosa.reshape"(%640) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_770 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %644 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%643, %642 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_770 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %645 = "tosa.reshape"(%644) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %646 = "tosa.add"(%545, %645) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %647 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_771 = arith.constant 2 : i32
    %648 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%646 : tensor<1x80x4096xf32>) outs(%647 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_771 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_772 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %649 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%648 : tensor<1x80x4096xf32>) outs(%cst_772 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %650 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %651 = "tosa.add"(%649, %650) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %652 = "tosa.rsqrt"(%651) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %653 = "tosa.mul"(%646, %652) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %654 = "tosa.reshape"(%extracted_slice_8) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %655 = "tosa.mul"(%654, %653) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %656 = tensor.empty() : tensor<4096x11008xf32>
    %657 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_130 : tensor<11008x4096xf32>) outs(%656 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %658 = "tosa.reshape"(%655) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_773 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %659 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%658, %657 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_773 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %660 = "tosa.reshape"(%659) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %661 = tensor.empty() : tensor<1x80x11008xf32>
    %662 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%660 : tensor<1x80x11008xf32>) outs(%661 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %663 = tensor.empty() : tensor<4096x11008xf32>
    %664 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_132 : tensor<11008x4096xf32>) outs(%663 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %665 = "tosa.reshape"(%655) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_774 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %666 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%665, %664 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_774 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %667 = "tosa.reshape"(%666) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %668 = "tosa.mul"(%662, %667) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %669 = tensor.empty() : tensor<11008x4096xf32>
    %670 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_134 : tensor<4096x11008xf32>) outs(%669 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %671 = "tosa.reshape"(%668) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_775 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %672 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%671, %670 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_775 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %673 = "tosa.reshape"(%672) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %674 = "tosa.add"(%646, %673) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %675 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_776 = arith.constant 2 : i32
    %676 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%674 : tensor<1x80x4096xf32>) outs(%675 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_776 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_777 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %677 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%676 : tensor<1x80x4096xf32>) outs(%cst_777 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %678 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %679 = "tosa.add"(%677, %678) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %680 = "tosa.rsqrt"(%679) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %681 = "tosa.mul"(%674, %680) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %682 = "tosa.reshape"(%extracted_slice_9) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %683 = "tosa.mul"(%682, %681) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %684 = tensor.empty() : tensor<4096x4096xf32>
    %685 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_136 : tensor<4096x4096xf32>) outs(%684 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %686 = "tosa.reshape"(%683) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_778 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %687 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%686, %685 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_778 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %688 = "tosa.reshape"(%687) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %689 = tensor.empty() : tensor<4096x4096xf32>
    %690 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_138 : tensor<4096x4096xf32>) outs(%689 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %691 = "tosa.reshape"(%683) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_779 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %692 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%691, %690 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_779 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %693 = "tosa.reshape"(%692) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %694 = tensor.empty() : tensor<4096x4096xf32>
    %695 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_140 : tensor<4096x4096xf32>) outs(%694 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %696 = "tosa.reshape"(%683) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_780 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %697 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%696, %695 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_780 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %698 = "tosa.reshape"(%697) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %699 = "tosa.reshape"(%688) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %700 = tensor.empty() : tensor<1x32x80x128xf32>
    %701 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%699 : tensor<1x80x32x128xf32>) outs(%700 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %702 = "tosa.reshape"(%693) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %703 = tensor.empty() : tensor<1x32x80x128xf32>
    %704 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%702 : tensor<1x80x32x128xf32>) outs(%703 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %705 = "tosa.reshape"(%698) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %706 = tensor.empty() : tensor<1x32x80x128xf32>
    %707 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%705 : tensor<1x80x32x128xf32>) outs(%706 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_781 = tensor.extract_slice %expanded_536[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_782 = tensor.extract_slice %extracted_slice_781[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_783 = tensor.extract_slice %extracted_slice_782[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_784 = tensor.extract_slice %expanded_538[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_785 = tensor.extract_slice %extracted_slice_784[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_786 = tensor.extract_slice %extracted_slice_785[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %708 = tensor.empty() : tensor<1x80x128xf32>
    %709 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_783 : tensor<1x1x80x128xf32>) outs(%708 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %710 = tensor.empty() : tensor<80x128xf32>
    %711 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%709 : tensor<1x80x128xf32>) outs(%710 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %712 = tensor.empty() : tensor<1x80x128xf32>
    %713 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_786 : tensor<1x1x80x128xf32>) outs(%712 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %714 = tensor.empty() : tensor<80x128xf32>
    %715 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%713 : tensor<1x80x128xf32>) outs(%714 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %716 = tensor.empty() : tensor<1x80x128xf32>
    %717 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%716 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %711[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %718 = "tosa.reshape"(%717) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %719 = tensor.empty() : tensor<1x80x128xf32>
    %720 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%719 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %715[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %721 = "tosa.reshape"(%720) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %722 = "tosa.mul"(%701, %718) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_787 = tensor.extract_slice %701[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_788 = tensor.extract_slice %701[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %723 = tensor.empty() : tensor<1x32x80x64xf32>
    %724 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_788 : tensor<1x32x80x64xf32>) outs(%723 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %725 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_789 = tensor.insert_slice %724 into %725[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_790 = tensor.insert_slice %extracted_slice_787 into %inserted_slice_789[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %726 = "tosa.mul"(%inserted_slice_790, %721) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %727 = "tosa.add"(%722, %726) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %728 = "tosa.mul"(%704, %718) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_791 = tensor.extract_slice %704[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_792 = tensor.extract_slice %704[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %729 = tensor.empty() : tensor<1x32x80x64xf32>
    %730 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_792 : tensor<1x32x80x64xf32>) outs(%729 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %731 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_793 = tensor.insert_slice %730 into %731[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_794 = tensor.insert_slice %extracted_slice_791 into %inserted_slice_793[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %732 = "tosa.mul"(%inserted_slice_794, %721) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %733 = "tosa.add"(%728, %732) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %734 = tensor.empty() : tensor<1x32x128x80xf32>
    %735 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%733 : tensor<1x32x80x128xf32>) outs(%734 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %736 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %737 = "tosa.add"(%727, %736) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %738 = "tosa.reshape"(%737) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %739 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %740 = "tosa.add"(%735, %739) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %741 = "tosa.reshape"(%740) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %742 = "tosa.matmul"(%738, %741) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %743 = "tosa.reshape"(%742) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %744 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %745 = "tosa.reciprocal"(%744) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %746 = "tosa.mul"(%743, %745) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %747 = "tosa.add"(%746, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %748 = tensor.empty() : tensor<1x32x80x1xf32>
    %749 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%748 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %750 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%747 : tensor<1x32x80x80xf32>) outs(%748 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %751 = tensor.empty() : tensor<1x32x80x80xf32>
    %752 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%747, %750 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%751 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %753 = tensor.empty() : tensor<1x32x80x1xf32>
    %754 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%753 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %755 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%752 : tensor<1x32x80x80xf32>) outs(%754 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %756 = tensor.empty() : tensor<1x32x80x80xf32>
    %757 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%752, %755 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%756 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %758 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %759 = "tosa.add"(%757, %758) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %760 = "tosa.reshape"(%759) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %761 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %762 = "tosa.add"(%707, %761) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %763 = "tosa.reshape"(%762) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %764 = "tosa.matmul"(%760, %763) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %765 = "tosa.reshape"(%764) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %766 = tensor.empty() : tensor<1x80x32x128xf32>
    %767 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%765 : tensor<1x32x80x128xf32>) outs(%766 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %768 = "tosa.identity"(%767) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %769 = "tosa.reshape"(%768) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %770 = tensor.empty() : tensor<4096x4096xf32>
    %771 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_142 : tensor<4096x4096xf32>) outs(%770 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %772 = "tosa.reshape"(%769) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_795 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %773 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%772, %771 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_795 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %774 = "tosa.reshape"(%773) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %775 = "tosa.add"(%674, %774) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %776 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_796 = arith.constant 2 : i32
    %777 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%775 : tensor<1x80x4096xf32>) outs(%776 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_796 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_797 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %778 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%777 : tensor<1x80x4096xf32>) outs(%cst_797 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %779 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %780 = "tosa.add"(%778, %779) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %781 = "tosa.rsqrt"(%780) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %782 = "tosa.mul"(%775, %781) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %783 = "tosa.reshape"(%extracted_slice_10) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %784 = "tosa.mul"(%783, %782) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %785 = tensor.empty() : tensor<4096x11008xf32>
    %786 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_144 : tensor<11008x4096xf32>) outs(%785 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %787 = "tosa.reshape"(%784) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_798 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %788 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%787, %786 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_798 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %789 = "tosa.reshape"(%788) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %790 = tensor.empty() : tensor<1x80x11008xf32>
    %791 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%789 : tensor<1x80x11008xf32>) outs(%790 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %792 = tensor.empty() : tensor<4096x11008xf32>
    %793 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_146 : tensor<11008x4096xf32>) outs(%792 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %794 = "tosa.reshape"(%784) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_799 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %795 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%794, %793 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_799 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %796 = "tosa.reshape"(%795) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %797 = "tosa.mul"(%791, %796) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %798 = tensor.empty() : tensor<11008x4096xf32>
    %799 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_148 : tensor<4096x11008xf32>) outs(%798 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %800 = "tosa.reshape"(%797) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_800 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %801 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%800, %799 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_800 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %802 = "tosa.reshape"(%801) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %803 = "tosa.add"(%775, %802) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %804 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_801 = arith.constant 2 : i32
    %805 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%803 : tensor<1x80x4096xf32>) outs(%804 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_801 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_802 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %806 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%805 : tensor<1x80x4096xf32>) outs(%cst_802 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %807 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %808 = "tosa.add"(%806, %807) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %809 = "tosa.rsqrt"(%808) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %810 = "tosa.mul"(%803, %809) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %811 = "tosa.reshape"(%extracted_slice_11) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %812 = "tosa.mul"(%811, %810) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %813 = tensor.empty() : tensor<4096x4096xf32>
    %814 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_150 : tensor<4096x4096xf32>) outs(%813 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %815 = "tosa.reshape"(%812) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_803 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %816 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%815, %814 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_803 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %817 = "tosa.reshape"(%816) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %818 = tensor.empty() : tensor<4096x4096xf32>
    %819 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_152 : tensor<4096x4096xf32>) outs(%818 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %820 = "tosa.reshape"(%812) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_804 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %821 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%820, %819 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_804 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %822 = "tosa.reshape"(%821) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %823 = tensor.empty() : tensor<4096x4096xf32>
    %824 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_154 : tensor<4096x4096xf32>) outs(%823 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %825 = "tosa.reshape"(%812) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_805 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %826 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%825, %824 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_805 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %827 = "tosa.reshape"(%826) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %828 = "tosa.reshape"(%817) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %829 = tensor.empty() : tensor<1x32x80x128xf32>
    %830 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%828 : tensor<1x80x32x128xf32>) outs(%829 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %831 = "tosa.reshape"(%822) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %832 = tensor.empty() : tensor<1x32x80x128xf32>
    %833 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%831 : tensor<1x80x32x128xf32>) outs(%832 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %834 = "tosa.reshape"(%827) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %835 = tensor.empty() : tensor<1x32x80x128xf32>
    %836 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%834 : tensor<1x80x32x128xf32>) outs(%835 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_806 = tensor.extract_slice %expanded_540[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_807 = tensor.extract_slice %extracted_slice_806[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_808 = tensor.extract_slice %extracted_slice_807[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_809 = tensor.extract_slice %expanded_542[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_810 = tensor.extract_slice %extracted_slice_809[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_811 = tensor.extract_slice %extracted_slice_810[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %837 = tensor.empty() : tensor<1x80x128xf32>
    %838 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_808 : tensor<1x1x80x128xf32>) outs(%837 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %839 = tensor.empty() : tensor<80x128xf32>
    %840 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%838 : tensor<1x80x128xf32>) outs(%839 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %841 = tensor.empty() : tensor<1x80x128xf32>
    %842 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_811 : tensor<1x1x80x128xf32>) outs(%841 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %843 = tensor.empty() : tensor<80x128xf32>
    %844 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%842 : tensor<1x80x128xf32>) outs(%843 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %845 = tensor.empty() : tensor<1x80x128xf32>
    %846 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%845 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %840[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %847 = "tosa.reshape"(%846) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %848 = tensor.empty() : tensor<1x80x128xf32>
    %849 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%848 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %844[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %850 = "tosa.reshape"(%849) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %851 = "tosa.mul"(%830, %847) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_812 = tensor.extract_slice %830[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_813 = tensor.extract_slice %830[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %852 = tensor.empty() : tensor<1x32x80x64xf32>
    %853 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_813 : tensor<1x32x80x64xf32>) outs(%852 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %854 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_814 = tensor.insert_slice %853 into %854[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_815 = tensor.insert_slice %extracted_slice_812 into %inserted_slice_814[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %855 = "tosa.mul"(%inserted_slice_815, %850) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %856 = "tosa.add"(%851, %855) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %857 = "tosa.mul"(%833, %847) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_816 = tensor.extract_slice %833[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_817 = tensor.extract_slice %833[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %858 = tensor.empty() : tensor<1x32x80x64xf32>
    %859 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_817 : tensor<1x32x80x64xf32>) outs(%858 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %860 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_818 = tensor.insert_slice %859 into %860[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_819 = tensor.insert_slice %extracted_slice_816 into %inserted_slice_818[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %861 = "tosa.mul"(%inserted_slice_819, %850) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %862 = "tosa.add"(%857, %861) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %863 = tensor.empty() : tensor<1x32x128x80xf32>
    %864 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%862 : tensor<1x32x80x128xf32>) outs(%863 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %865 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %866 = "tosa.add"(%856, %865) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %867 = "tosa.reshape"(%866) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %868 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %869 = "tosa.add"(%864, %868) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %870 = "tosa.reshape"(%869) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %871 = "tosa.matmul"(%867, %870) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %872 = "tosa.reshape"(%871) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %873 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %874 = "tosa.reciprocal"(%873) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %875 = "tosa.mul"(%872, %874) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %876 = "tosa.add"(%875, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %877 = tensor.empty() : tensor<1x32x80x1xf32>
    %878 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%877 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %879 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%876 : tensor<1x32x80x80xf32>) outs(%877 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %880 = tensor.empty() : tensor<1x32x80x80xf32>
    %881 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%876, %879 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%880 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %882 = tensor.empty() : tensor<1x32x80x1xf32>
    %883 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%882 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %884 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%881 : tensor<1x32x80x80xf32>) outs(%883 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %885 = tensor.empty() : tensor<1x32x80x80xf32>
    %886 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%881, %884 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%885 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %887 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %888 = "tosa.add"(%886, %887) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %889 = "tosa.reshape"(%888) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %890 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %891 = "tosa.add"(%836, %890) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %892 = "tosa.reshape"(%891) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %893 = "tosa.matmul"(%889, %892) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %894 = "tosa.reshape"(%893) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %895 = tensor.empty() : tensor<1x80x32x128xf32>
    %896 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%894 : tensor<1x32x80x128xf32>) outs(%895 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %897 = "tosa.identity"(%896) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %898 = "tosa.reshape"(%897) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %899 = tensor.empty() : tensor<4096x4096xf32>
    %900 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_156 : tensor<4096x4096xf32>) outs(%899 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %901 = "tosa.reshape"(%898) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_820 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %902 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%901, %900 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_820 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %903 = "tosa.reshape"(%902) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %904 = "tosa.add"(%803, %903) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %905 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_821 = arith.constant 2 : i32
    %906 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%904 : tensor<1x80x4096xf32>) outs(%905 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_821 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_822 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %907 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%906 : tensor<1x80x4096xf32>) outs(%cst_822 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %908 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %909 = "tosa.add"(%907, %908) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %910 = "tosa.rsqrt"(%909) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %911 = "tosa.mul"(%904, %910) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %912 = "tosa.reshape"(%extracted_slice_12) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %913 = "tosa.mul"(%912, %911) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %914 = tensor.empty() : tensor<4096x11008xf32>
    %915 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_158 : tensor<11008x4096xf32>) outs(%914 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %916 = "tosa.reshape"(%913) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_823 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %917 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%916, %915 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_823 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %918 = "tosa.reshape"(%917) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %919 = tensor.empty() : tensor<1x80x11008xf32>
    %920 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%918 : tensor<1x80x11008xf32>) outs(%919 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %921 = tensor.empty() : tensor<4096x11008xf32>
    %922 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_160 : tensor<11008x4096xf32>) outs(%921 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %923 = "tosa.reshape"(%913) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_824 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %924 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%923, %922 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_824 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %925 = "tosa.reshape"(%924) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %926 = "tosa.mul"(%920, %925) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %927 = tensor.empty() : tensor<11008x4096xf32>
    %928 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_162 : tensor<4096x11008xf32>) outs(%927 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %929 = "tosa.reshape"(%926) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_825 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %930 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%929, %928 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_825 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %931 = "tosa.reshape"(%930) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %932 = "tosa.add"(%904, %931) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %933 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_826 = arith.constant 2 : i32
    %934 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%932 : tensor<1x80x4096xf32>) outs(%933 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_826 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_827 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %935 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%934 : tensor<1x80x4096xf32>) outs(%cst_827 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %936 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %937 = "tosa.add"(%935, %936) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %938 = "tosa.rsqrt"(%937) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %939 = "tosa.mul"(%932, %938) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %940 = "tosa.reshape"(%extracted_slice_13) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %941 = "tosa.mul"(%940, %939) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %942 = tensor.empty() : tensor<4096x4096xf32>
    %943 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_164 : tensor<4096x4096xf32>) outs(%942 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %944 = "tosa.reshape"(%941) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_828 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %945 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%944, %943 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_828 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %946 = "tosa.reshape"(%945) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %947 = tensor.empty() : tensor<4096x4096xf32>
    %948 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_166 : tensor<4096x4096xf32>) outs(%947 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %949 = "tosa.reshape"(%941) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_829 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %950 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%949, %948 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_829 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %951 = "tosa.reshape"(%950) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %952 = tensor.empty() : tensor<4096x4096xf32>
    %953 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_168 : tensor<4096x4096xf32>) outs(%952 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %954 = "tosa.reshape"(%941) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_830 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %955 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%954, %953 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_830 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %956 = "tosa.reshape"(%955) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %957 = "tosa.reshape"(%946) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %958 = tensor.empty() : tensor<1x32x80x128xf32>
    %959 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%957 : tensor<1x80x32x128xf32>) outs(%958 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %960 = "tosa.reshape"(%951) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %961 = tensor.empty() : tensor<1x32x80x128xf32>
    %962 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%960 : tensor<1x80x32x128xf32>) outs(%961 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %963 = "tosa.reshape"(%956) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %964 = tensor.empty() : tensor<1x32x80x128xf32>
    %965 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%963 : tensor<1x80x32x128xf32>) outs(%964 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_831 = tensor.extract_slice %expanded_544[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_832 = tensor.extract_slice %extracted_slice_831[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_833 = tensor.extract_slice %extracted_slice_832[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_834 = tensor.extract_slice %expanded_546[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_835 = tensor.extract_slice %extracted_slice_834[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_836 = tensor.extract_slice %extracted_slice_835[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %966 = tensor.empty() : tensor<1x80x128xf32>
    %967 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_833 : tensor<1x1x80x128xf32>) outs(%966 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %968 = tensor.empty() : tensor<80x128xf32>
    %969 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%967 : tensor<1x80x128xf32>) outs(%968 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %970 = tensor.empty() : tensor<1x80x128xf32>
    %971 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_836 : tensor<1x1x80x128xf32>) outs(%970 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %972 = tensor.empty() : tensor<80x128xf32>
    %973 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%971 : tensor<1x80x128xf32>) outs(%972 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %974 = tensor.empty() : tensor<1x80x128xf32>
    %975 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%974 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %969[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %976 = "tosa.reshape"(%975) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %977 = tensor.empty() : tensor<1x80x128xf32>
    %978 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%977 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %973[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %979 = "tosa.reshape"(%978) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %980 = "tosa.mul"(%959, %976) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_837 = tensor.extract_slice %959[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_838 = tensor.extract_slice %959[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %981 = tensor.empty() : tensor<1x32x80x64xf32>
    %982 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_838 : tensor<1x32x80x64xf32>) outs(%981 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %983 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_839 = tensor.insert_slice %982 into %983[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_840 = tensor.insert_slice %extracted_slice_837 into %inserted_slice_839[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %984 = "tosa.mul"(%inserted_slice_840, %979) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %985 = "tosa.add"(%980, %984) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %986 = "tosa.mul"(%962, %976) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_841 = tensor.extract_slice %962[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_842 = tensor.extract_slice %962[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %987 = tensor.empty() : tensor<1x32x80x64xf32>
    %988 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_842 : tensor<1x32x80x64xf32>) outs(%987 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %989 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_843 = tensor.insert_slice %988 into %989[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_844 = tensor.insert_slice %extracted_slice_841 into %inserted_slice_843[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %990 = "tosa.mul"(%inserted_slice_844, %979) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %991 = "tosa.add"(%986, %990) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %992 = tensor.empty() : tensor<1x32x128x80xf32>
    %993 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%991 : tensor<1x32x80x128xf32>) outs(%992 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %994 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %995 = "tosa.add"(%985, %994) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %996 = "tosa.reshape"(%995) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %997 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %998 = "tosa.add"(%993, %997) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %999 = "tosa.reshape"(%998) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %1000 = "tosa.matmul"(%996, %999) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %1001 = "tosa.reshape"(%1000) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1002 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1003 = "tosa.reciprocal"(%1002) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1004 = "tosa.mul"(%1001, %1003) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1005 = "tosa.add"(%1004, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1006 = tensor.empty() : tensor<1x32x80x1xf32>
    %1007 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1006 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1008 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1005 : tensor<1x32x80x80xf32>) outs(%1006 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1009 = tensor.empty() : tensor<1x32x80x80xf32>
    %1010 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1005, %1008 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1009 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %1011 = tensor.empty() : tensor<1x32x80x1xf32>
    %1012 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1011 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1013 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1010 : tensor<1x32x80x80xf32>) outs(%1012 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1014 = tensor.empty() : tensor<1x32x80x80xf32>
    %1015 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1010, %1013 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1014 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %1016 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1017 = "tosa.add"(%1015, %1016) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1018 = "tosa.reshape"(%1017) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %1019 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1020 = "tosa.add"(%965, %1019) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1021 = "tosa.reshape"(%1020) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1022 = "tosa.matmul"(%1018, %1021) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %1023 = "tosa.reshape"(%1022) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1024 = tensor.empty() : tensor<1x80x32x128xf32>
    %1025 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1023 : tensor<1x32x80x128xf32>) outs(%1024 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %1026 = "tosa.identity"(%1025) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %1027 = "tosa.reshape"(%1026) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %1028 = tensor.empty() : tensor<4096x4096xf32>
    %1029 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_170 : tensor<4096x4096xf32>) outs(%1028 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1030 = "tosa.reshape"(%1027) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_845 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1031 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1030, %1029 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_845 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1032 = "tosa.reshape"(%1031) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1033 = "tosa.add"(%932, %1032) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1034 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_846 = arith.constant 2 : i32
    %1035 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1033 : tensor<1x80x4096xf32>) outs(%1034 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_846 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_847 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1036 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1035 : tensor<1x80x4096xf32>) outs(%cst_847 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1037 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1038 = "tosa.add"(%1036, %1037) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1039 = "tosa.rsqrt"(%1038) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1040 = "tosa.mul"(%1033, %1039) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1041 = "tosa.reshape"(%extracted_slice_14) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1042 = "tosa.mul"(%1041, %1040) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1043 = tensor.empty() : tensor<4096x11008xf32>
    %1044 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_172 : tensor<11008x4096xf32>) outs(%1043 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1045 = "tosa.reshape"(%1042) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_848 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1046 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1045, %1044 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_848 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1047 = "tosa.reshape"(%1046) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1048 = tensor.empty() : tensor<1x80x11008xf32>
    %1049 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1047 : tensor<1x80x11008xf32>) outs(%1048 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %1050 = tensor.empty() : tensor<4096x11008xf32>
    %1051 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_174 : tensor<11008x4096xf32>) outs(%1050 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1052 = "tosa.reshape"(%1042) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_849 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1053 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1052, %1051 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_849 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1054 = "tosa.reshape"(%1053) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1055 = "tosa.mul"(%1049, %1054) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %1056 = tensor.empty() : tensor<11008x4096xf32>
    %1057 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_176 : tensor<4096x11008xf32>) outs(%1056 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %1058 = "tosa.reshape"(%1055) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_850 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1059 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1058, %1057 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_850 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1060 = "tosa.reshape"(%1059) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1061 = "tosa.add"(%1033, %1060) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1062 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_851 = arith.constant 2 : i32
    %1063 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1061 : tensor<1x80x4096xf32>) outs(%1062 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_851 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_852 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1064 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1063 : tensor<1x80x4096xf32>) outs(%cst_852 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1065 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1066 = "tosa.add"(%1064, %1065) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1067 = "tosa.rsqrt"(%1066) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1068 = "tosa.mul"(%1061, %1067) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1069 = "tosa.reshape"(%extracted_slice_15) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1070 = "tosa.mul"(%1069, %1068) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1071 = tensor.empty() : tensor<4096x4096xf32>
    %1072 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_178 : tensor<4096x4096xf32>) outs(%1071 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1073 = "tosa.reshape"(%1070) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_853 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1074 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1073, %1072 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_853 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1075 = "tosa.reshape"(%1074) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1076 = tensor.empty() : tensor<4096x4096xf32>
    %1077 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_180 : tensor<4096x4096xf32>) outs(%1076 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1078 = "tosa.reshape"(%1070) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_854 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1079 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1078, %1077 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_854 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1080 = "tosa.reshape"(%1079) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1081 = tensor.empty() : tensor<4096x4096xf32>
    %1082 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_182 : tensor<4096x4096xf32>) outs(%1081 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1083 = "tosa.reshape"(%1070) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_855 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1084 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1083, %1082 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_855 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1085 = "tosa.reshape"(%1084) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1086 = "tosa.reshape"(%1075) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1087 = tensor.empty() : tensor<1x32x80x128xf32>
    %1088 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1086 : tensor<1x80x32x128xf32>) outs(%1087 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1089 = "tosa.reshape"(%1080) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1090 = tensor.empty() : tensor<1x32x80x128xf32>
    %1091 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1089 : tensor<1x80x32x128xf32>) outs(%1090 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1092 = "tosa.reshape"(%1085) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1093 = tensor.empty() : tensor<1x32x80x128xf32>
    %1094 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1092 : tensor<1x80x32x128xf32>) outs(%1093 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_856 = tensor.extract_slice %expanded_548[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_857 = tensor.extract_slice %extracted_slice_856[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_858 = tensor.extract_slice %extracted_slice_857[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_859 = tensor.extract_slice %expanded_550[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_860 = tensor.extract_slice %extracted_slice_859[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_861 = tensor.extract_slice %extracted_slice_860[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %1095 = tensor.empty() : tensor<1x80x128xf32>
    %1096 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_858 : tensor<1x1x80x128xf32>) outs(%1095 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1097 = tensor.empty() : tensor<80x128xf32>
    %1098 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1096 : tensor<1x80x128xf32>) outs(%1097 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1099 = tensor.empty() : tensor<1x80x128xf32>
    %1100 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_861 : tensor<1x1x80x128xf32>) outs(%1099 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1101 = tensor.empty() : tensor<80x128xf32>
    %1102 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1100 : tensor<1x80x128xf32>) outs(%1101 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1103 = tensor.empty() : tensor<1x80x128xf32>
    %1104 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1103 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1098[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1105 = "tosa.reshape"(%1104) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1106 = tensor.empty() : tensor<1x80x128xf32>
    %1107 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1106 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1102[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1108 = "tosa.reshape"(%1107) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1109 = "tosa.mul"(%1088, %1105) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_862 = tensor.extract_slice %1088[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_863 = tensor.extract_slice %1088[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1110 = tensor.empty() : tensor<1x32x80x64xf32>
    %1111 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_863 : tensor<1x32x80x64xf32>) outs(%1110 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1112 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_864 = tensor.insert_slice %1111 into %1112[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_865 = tensor.insert_slice %extracted_slice_862 into %inserted_slice_864[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1113 = "tosa.mul"(%inserted_slice_865, %1108) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1114 = "tosa.add"(%1109, %1113) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1115 = "tosa.mul"(%1091, %1105) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_866 = tensor.extract_slice %1091[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_867 = tensor.extract_slice %1091[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1116 = tensor.empty() : tensor<1x32x80x64xf32>
    %1117 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_867 : tensor<1x32x80x64xf32>) outs(%1116 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1118 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_868 = tensor.insert_slice %1117 into %1118[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_869 = tensor.insert_slice %extracted_slice_866 into %inserted_slice_868[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1119 = "tosa.mul"(%inserted_slice_869, %1108) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1120 = "tosa.add"(%1115, %1119) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1121 = tensor.empty() : tensor<1x32x128x80xf32>
    %1122 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1120 : tensor<1x32x80x128xf32>) outs(%1121 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %1123 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1124 = "tosa.add"(%1114, %1123) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1125 = "tosa.reshape"(%1124) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1126 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %1127 = "tosa.add"(%1122, %1126) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %1128 = "tosa.reshape"(%1127) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %1129 = "tosa.matmul"(%1125, %1128) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %1130 = "tosa.reshape"(%1129) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1131 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1132 = "tosa.reciprocal"(%1131) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1133 = "tosa.mul"(%1130, %1132) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1134 = "tosa.add"(%1133, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1135 = tensor.empty() : tensor<1x32x80x1xf32>
    %1136 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1135 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1137 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1134 : tensor<1x32x80x80xf32>) outs(%1135 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1138 = tensor.empty() : tensor<1x32x80x80xf32>
    %1139 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1134, %1137 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1138 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %1140 = tensor.empty() : tensor<1x32x80x1xf32>
    %1141 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1140 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1142 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1139 : tensor<1x32x80x80xf32>) outs(%1141 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1143 = tensor.empty() : tensor<1x32x80x80xf32>
    %1144 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1139, %1142 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1143 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %1145 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1146 = "tosa.add"(%1144, %1145) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1147 = "tosa.reshape"(%1146) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %1148 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1149 = "tosa.add"(%1094, %1148) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1150 = "tosa.reshape"(%1149) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1151 = "tosa.matmul"(%1147, %1150) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %1152 = "tosa.reshape"(%1151) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1153 = tensor.empty() : tensor<1x80x32x128xf32>
    %1154 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1152 : tensor<1x32x80x128xf32>) outs(%1153 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %1155 = "tosa.identity"(%1154) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %1156 = "tosa.reshape"(%1155) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %1157 = tensor.empty() : tensor<4096x4096xf32>
    %1158 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_184 : tensor<4096x4096xf32>) outs(%1157 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1159 = "tosa.reshape"(%1156) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_870 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1160 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1159, %1158 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_870 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1161 = "tosa.reshape"(%1160) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1162 = "tosa.add"(%1061, %1161) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1163 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_871 = arith.constant 2 : i32
    %1164 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1162 : tensor<1x80x4096xf32>) outs(%1163 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_871 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_872 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1165 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1164 : tensor<1x80x4096xf32>) outs(%cst_872 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1166 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1167 = "tosa.add"(%1165, %1166) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1168 = "tosa.rsqrt"(%1167) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1169 = "tosa.mul"(%1162, %1168) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1170 = "tosa.reshape"(%extracted_slice_16) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1171 = "tosa.mul"(%1170, %1169) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1172 = tensor.empty() : tensor<4096x11008xf32>
    %1173 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_186 : tensor<11008x4096xf32>) outs(%1172 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1174 = "tosa.reshape"(%1171) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_873 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1175 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1174, %1173 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_873 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1176 = "tosa.reshape"(%1175) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1177 = tensor.empty() : tensor<1x80x11008xf32>
    %1178 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1176 : tensor<1x80x11008xf32>) outs(%1177 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %1179 = tensor.empty() : tensor<4096x11008xf32>
    %1180 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_188 : tensor<11008x4096xf32>) outs(%1179 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1181 = "tosa.reshape"(%1171) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_874 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1182 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1181, %1180 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_874 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1183 = "tosa.reshape"(%1182) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1184 = "tosa.mul"(%1178, %1183) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %1185 = tensor.empty() : tensor<11008x4096xf32>
    %1186 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_190 : tensor<4096x11008xf32>) outs(%1185 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %1187 = "tosa.reshape"(%1184) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_875 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1188 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1187, %1186 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_875 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1189 = "tosa.reshape"(%1188) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1190 = "tosa.add"(%1162, %1189) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1191 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_876 = arith.constant 2 : i32
    %1192 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1190 : tensor<1x80x4096xf32>) outs(%1191 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_876 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_877 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1193 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1192 : tensor<1x80x4096xf32>) outs(%cst_877 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1194 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1195 = "tosa.add"(%1193, %1194) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1196 = "tosa.rsqrt"(%1195) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1197 = "tosa.mul"(%1190, %1196) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1198 = "tosa.reshape"(%extracted_slice_17) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1199 = "tosa.mul"(%1198, %1197) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1200 = tensor.empty() : tensor<4096x4096xf32>
    %1201 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_192 : tensor<4096x4096xf32>) outs(%1200 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1202 = "tosa.reshape"(%1199) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_878 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1203 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1202, %1201 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_878 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1204 = "tosa.reshape"(%1203) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1205 = tensor.empty() : tensor<4096x4096xf32>
    %1206 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_194 : tensor<4096x4096xf32>) outs(%1205 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1207 = "tosa.reshape"(%1199) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_879 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1208 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1207, %1206 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_879 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1209 = "tosa.reshape"(%1208) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1210 = tensor.empty() : tensor<4096x4096xf32>
    %1211 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_196 : tensor<4096x4096xf32>) outs(%1210 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1212 = "tosa.reshape"(%1199) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_880 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1213 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1212, %1211 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_880 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1214 = "tosa.reshape"(%1213) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1215 = "tosa.reshape"(%1204) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1216 = tensor.empty() : tensor<1x32x80x128xf32>
    %1217 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1215 : tensor<1x80x32x128xf32>) outs(%1216 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1218 = "tosa.reshape"(%1209) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1219 = tensor.empty() : tensor<1x32x80x128xf32>
    %1220 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1218 : tensor<1x80x32x128xf32>) outs(%1219 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1221 = "tosa.reshape"(%1214) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1222 = tensor.empty() : tensor<1x32x80x128xf32>
    %1223 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1221 : tensor<1x80x32x128xf32>) outs(%1222 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_881 = tensor.extract_slice %expanded_552[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_882 = tensor.extract_slice %extracted_slice_881[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_883 = tensor.extract_slice %extracted_slice_882[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_884 = tensor.extract_slice %expanded_554[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_885 = tensor.extract_slice %extracted_slice_884[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_886 = tensor.extract_slice %extracted_slice_885[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %1224 = tensor.empty() : tensor<1x80x128xf32>
    %1225 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_883 : tensor<1x1x80x128xf32>) outs(%1224 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1226 = tensor.empty() : tensor<80x128xf32>
    %1227 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1225 : tensor<1x80x128xf32>) outs(%1226 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1228 = tensor.empty() : tensor<1x80x128xf32>
    %1229 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_886 : tensor<1x1x80x128xf32>) outs(%1228 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1230 = tensor.empty() : tensor<80x128xf32>
    %1231 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1229 : tensor<1x80x128xf32>) outs(%1230 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1232 = tensor.empty() : tensor<1x80x128xf32>
    %1233 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1232 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1227[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1234 = "tosa.reshape"(%1233) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1235 = tensor.empty() : tensor<1x80x128xf32>
    %1236 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1235 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1231[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1237 = "tosa.reshape"(%1236) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1238 = "tosa.mul"(%1217, %1234) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_887 = tensor.extract_slice %1217[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_888 = tensor.extract_slice %1217[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1239 = tensor.empty() : tensor<1x32x80x64xf32>
    %1240 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_888 : tensor<1x32x80x64xf32>) outs(%1239 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1241 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_889 = tensor.insert_slice %1240 into %1241[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_890 = tensor.insert_slice %extracted_slice_887 into %inserted_slice_889[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1242 = "tosa.mul"(%inserted_slice_890, %1237) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1243 = "tosa.add"(%1238, %1242) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1244 = "tosa.mul"(%1220, %1234) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_891 = tensor.extract_slice %1220[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_892 = tensor.extract_slice %1220[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1245 = tensor.empty() : tensor<1x32x80x64xf32>
    %1246 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_892 : tensor<1x32x80x64xf32>) outs(%1245 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1247 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_893 = tensor.insert_slice %1246 into %1247[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_894 = tensor.insert_slice %extracted_slice_891 into %inserted_slice_893[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1248 = "tosa.mul"(%inserted_slice_894, %1237) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1249 = "tosa.add"(%1244, %1248) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1250 = tensor.empty() : tensor<1x32x128x80xf32>
    %1251 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1249 : tensor<1x32x80x128xf32>) outs(%1250 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %1252 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1253 = "tosa.add"(%1243, %1252) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1254 = "tosa.reshape"(%1253) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1255 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %1256 = "tosa.add"(%1251, %1255) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %1257 = "tosa.reshape"(%1256) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %1258 = "tosa.matmul"(%1254, %1257) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %1259 = "tosa.reshape"(%1258) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1260 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1261 = "tosa.reciprocal"(%1260) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1262 = "tosa.mul"(%1259, %1261) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1263 = "tosa.add"(%1262, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1264 = tensor.empty() : tensor<1x32x80x1xf32>
    %1265 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1264 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1266 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1263 : tensor<1x32x80x80xf32>) outs(%1264 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1267 = tensor.empty() : tensor<1x32x80x80xf32>
    %1268 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1263, %1266 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1267 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %1269 = tensor.empty() : tensor<1x32x80x1xf32>
    %1270 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1269 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1271 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1268 : tensor<1x32x80x80xf32>) outs(%1270 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1272 = tensor.empty() : tensor<1x32x80x80xf32>
    %1273 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1268, %1271 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1272 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %1274 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1275 = "tosa.add"(%1273, %1274) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1276 = "tosa.reshape"(%1275) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %1277 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1278 = "tosa.add"(%1223, %1277) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1279 = "tosa.reshape"(%1278) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1280 = "tosa.matmul"(%1276, %1279) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %1281 = "tosa.reshape"(%1280) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1282 = tensor.empty() : tensor<1x80x32x128xf32>
    %1283 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1281 : tensor<1x32x80x128xf32>) outs(%1282 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %1284 = "tosa.identity"(%1283) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %1285 = "tosa.reshape"(%1284) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %1286 = tensor.empty() : tensor<4096x4096xf32>
    %1287 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_198 : tensor<4096x4096xf32>) outs(%1286 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1288 = "tosa.reshape"(%1285) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_895 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1289 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1288, %1287 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_895 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1290 = "tosa.reshape"(%1289) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1291 = "tosa.add"(%1190, %1290) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1292 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_896 = arith.constant 2 : i32
    %1293 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1291 : tensor<1x80x4096xf32>) outs(%1292 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_896 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_897 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1294 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1293 : tensor<1x80x4096xf32>) outs(%cst_897 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1295 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1296 = "tosa.add"(%1294, %1295) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1297 = "tosa.rsqrt"(%1296) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1298 = "tosa.mul"(%1291, %1297) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1299 = "tosa.reshape"(%extracted_slice_18) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1300 = "tosa.mul"(%1299, %1298) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1301 = tensor.empty() : tensor<4096x11008xf32>
    %1302 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_200 : tensor<11008x4096xf32>) outs(%1301 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1303 = "tosa.reshape"(%1300) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_898 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1304 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1303, %1302 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_898 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1305 = "tosa.reshape"(%1304) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1306 = tensor.empty() : tensor<1x80x11008xf32>
    %1307 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1305 : tensor<1x80x11008xf32>) outs(%1306 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %1308 = tensor.empty() : tensor<4096x11008xf32>
    %1309 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_202 : tensor<11008x4096xf32>) outs(%1308 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1310 = "tosa.reshape"(%1300) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_899 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1311 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1310, %1309 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_899 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1312 = "tosa.reshape"(%1311) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1313 = "tosa.mul"(%1307, %1312) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %1314 = tensor.empty() : tensor<11008x4096xf32>
    %1315 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_204 : tensor<4096x11008xf32>) outs(%1314 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %1316 = "tosa.reshape"(%1313) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_900 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1317 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1316, %1315 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_900 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1318 = "tosa.reshape"(%1317) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1319 = "tosa.add"(%1291, %1318) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1320 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_901 = arith.constant 2 : i32
    %1321 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1319 : tensor<1x80x4096xf32>) outs(%1320 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_901 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_902 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1322 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1321 : tensor<1x80x4096xf32>) outs(%cst_902 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1323 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1324 = "tosa.add"(%1322, %1323) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1325 = "tosa.rsqrt"(%1324) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1326 = "tosa.mul"(%1319, %1325) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1327 = "tosa.reshape"(%extracted_slice_19) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1328 = "tosa.mul"(%1327, %1326) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1329 = tensor.empty() : tensor<4096x4096xf32>
    %1330 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_206 : tensor<4096x4096xf32>) outs(%1329 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1331 = "tosa.reshape"(%1328) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_903 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1332 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1331, %1330 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_903 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1333 = "tosa.reshape"(%1332) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1334 = tensor.empty() : tensor<4096x4096xf32>
    %1335 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_208 : tensor<4096x4096xf32>) outs(%1334 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1336 = "tosa.reshape"(%1328) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_904 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1337 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1336, %1335 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_904 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1338 = "tosa.reshape"(%1337) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1339 = tensor.empty() : tensor<4096x4096xf32>
    %1340 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_210 : tensor<4096x4096xf32>) outs(%1339 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1341 = "tosa.reshape"(%1328) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_905 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1342 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1341, %1340 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_905 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1343 = "tosa.reshape"(%1342) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1344 = "tosa.reshape"(%1333) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1345 = tensor.empty() : tensor<1x32x80x128xf32>
    %1346 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1344 : tensor<1x80x32x128xf32>) outs(%1345 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1347 = "tosa.reshape"(%1338) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1348 = tensor.empty() : tensor<1x32x80x128xf32>
    %1349 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1347 : tensor<1x80x32x128xf32>) outs(%1348 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1350 = "tosa.reshape"(%1343) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1351 = tensor.empty() : tensor<1x32x80x128xf32>
    %1352 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1350 : tensor<1x80x32x128xf32>) outs(%1351 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_906 = tensor.extract_slice %expanded_556[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_907 = tensor.extract_slice %extracted_slice_906[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_908 = tensor.extract_slice %extracted_slice_907[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_909 = tensor.extract_slice %expanded_558[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_910 = tensor.extract_slice %extracted_slice_909[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_911 = tensor.extract_slice %extracted_slice_910[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %1353 = tensor.empty() : tensor<1x80x128xf32>
    %1354 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_908 : tensor<1x1x80x128xf32>) outs(%1353 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1355 = tensor.empty() : tensor<80x128xf32>
    %1356 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1354 : tensor<1x80x128xf32>) outs(%1355 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1357 = tensor.empty() : tensor<1x80x128xf32>
    %1358 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_911 : tensor<1x1x80x128xf32>) outs(%1357 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1359 = tensor.empty() : tensor<80x128xf32>
    %1360 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1358 : tensor<1x80x128xf32>) outs(%1359 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1361 = tensor.empty() : tensor<1x80x128xf32>
    %1362 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1361 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1356[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1363 = "tosa.reshape"(%1362) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1364 = tensor.empty() : tensor<1x80x128xf32>
    %1365 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1364 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1360[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1366 = "tosa.reshape"(%1365) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1367 = "tosa.mul"(%1346, %1363) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_912 = tensor.extract_slice %1346[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_913 = tensor.extract_slice %1346[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1368 = tensor.empty() : tensor<1x32x80x64xf32>
    %1369 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_913 : tensor<1x32x80x64xf32>) outs(%1368 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1370 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_914 = tensor.insert_slice %1369 into %1370[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_915 = tensor.insert_slice %extracted_slice_912 into %inserted_slice_914[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1371 = "tosa.mul"(%inserted_slice_915, %1366) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1372 = "tosa.add"(%1367, %1371) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1373 = "tosa.mul"(%1349, %1363) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_916 = tensor.extract_slice %1349[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_917 = tensor.extract_slice %1349[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1374 = tensor.empty() : tensor<1x32x80x64xf32>
    %1375 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_917 : tensor<1x32x80x64xf32>) outs(%1374 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1376 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_918 = tensor.insert_slice %1375 into %1376[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_919 = tensor.insert_slice %extracted_slice_916 into %inserted_slice_918[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1377 = "tosa.mul"(%inserted_slice_919, %1366) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1378 = "tosa.add"(%1373, %1377) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1379 = tensor.empty() : tensor<1x32x128x80xf32>
    %1380 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1378 : tensor<1x32x80x128xf32>) outs(%1379 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %1381 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1382 = "tosa.add"(%1372, %1381) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1383 = "tosa.reshape"(%1382) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1384 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %1385 = "tosa.add"(%1380, %1384) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %1386 = "tosa.reshape"(%1385) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %1387 = "tosa.matmul"(%1383, %1386) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %1388 = "tosa.reshape"(%1387) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1389 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1390 = "tosa.reciprocal"(%1389) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1391 = "tosa.mul"(%1388, %1390) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1392 = "tosa.add"(%1391, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1393 = tensor.empty() : tensor<1x32x80x1xf32>
    %1394 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1393 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1395 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1392 : tensor<1x32x80x80xf32>) outs(%1393 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1396 = tensor.empty() : tensor<1x32x80x80xf32>
    %1397 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1392, %1395 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1396 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %1398 = tensor.empty() : tensor<1x32x80x1xf32>
    %1399 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1398 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1400 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1397 : tensor<1x32x80x80xf32>) outs(%1399 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1401 = tensor.empty() : tensor<1x32x80x80xf32>
    %1402 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1397, %1400 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1401 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %1403 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1404 = "tosa.add"(%1402, %1403) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1405 = "tosa.reshape"(%1404) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %1406 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1407 = "tosa.add"(%1352, %1406) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1408 = "tosa.reshape"(%1407) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1409 = "tosa.matmul"(%1405, %1408) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %1410 = "tosa.reshape"(%1409) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1411 = tensor.empty() : tensor<1x80x32x128xf32>
    %1412 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1410 : tensor<1x32x80x128xf32>) outs(%1411 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %1413 = "tosa.identity"(%1412) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %1414 = "tosa.reshape"(%1413) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %1415 = tensor.empty() : tensor<4096x4096xf32>
    %1416 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_212 : tensor<4096x4096xf32>) outs(%1415 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1417 = "tosa.reshape"(%1414) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_920 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1418 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1417, %1416 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_920 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1419 = "tosa.reshape"(%1418) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1420 = "tosa.add"(%1319, %1419) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1421 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_921 = arith.constant 2 : i32
    %1422 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1420 : tensor<1x80x4096xf32>) outs(%1421 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_921 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_922 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1423 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1422 : tensor<1x80x4096xf32>) outs(%cst_922 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1424 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1425 = "tosa.add"(%1423, %1424) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1426 = "tosa.rsqrt"(%1425) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1427 = "tosa.mul"(%1420, %1426) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1428 = "tosa.reshape"(%extracted_slice_20) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1429 = "tosa.mul"(%1428, %1427) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1430 = tensor.empty() : tensor<4096x11008xf32>
    %1431 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_214 : tensor<11008x4096xf32>) outs(%1430 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1432 = "tosa.reshape"(%1429) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_923 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1433 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1432, %1431 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_923 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1434 = "tosa.reshape"(%1433) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1435 = tensor.empty() : tensor<1x80x11008xf32>
    %1436 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1434 : tensor<1x80x11008xf32>) outs(%1435 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %1437 = tensor.empty() : tensor<4096x11008xf32>
    %1438 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_216 : tensor<11008x4096xf32>) outs(%1437 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1439 = "tosa.reshape"(%1429) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_924 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1440 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1439, %1438 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_924 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1441 = "tosa.reshape"(%1440) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1442 = "tosa.mul"(%1436, %1441) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %1443 = tensor.empty() : tensor<11008x4096xf32>
    %1444 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_218 : tensor<4096x11008xf32>) outs(%1443 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %1445 = "tosa.reshape"(%1442) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_925 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1446 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1445, %1444 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_925 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1447 = "tosa.reshape"(%1446) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1448 = "tosa.add"(%1420, %1447) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1449 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_926 = arith.constant 2 : i32
    %1450 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1448 : tensor<1x80x4096xf32>) outs(%1449 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_926 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_927 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1451 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1450 : tensor<1x80x4096xf32>) outs(%cst_927 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1452 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1453 = "tosa.add"(%1451, %1452) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1454 = "tosa.rsqrt"(%1453) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1455 = "tosa.mul"(%1448, %1454) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1456 = "tosa.reshape"(%extracted_slice_21) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1457 = "tosa.mul"(%1456, %1455) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1458 = tensor.empty() : tensor<4096x4096xf32>
    %1459 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_220 : tensor<4096x4096xf32>) outs(%1458 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1460 = "tosa.reshape"(%1457) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_928 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1461 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1460, %1459 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_928 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1462 = "tosa.reshape"(%1461) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1463 = tensor.empty() : tensor<4096x4096xf32>
    %1464 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_222 : tensor<4096x4096xf32>) outs(%1463 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1465 = "tosa.reshape"(%1457) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_929 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1466 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1465, %1464 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_929 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1467 = "tosa.reshape"(%1466) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1468 = tensor.empty() : tensor<4096x4096xf32>
    %1469 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_224 : tensor<4096x4096xf32>) outs(%1468 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1470 = "tosa.reshape"(%1457) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_930 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1471 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1470, %1469 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_930 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1472 = "tosa.reshape"(%1471) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1473 = "tosa.reshape"(%1462) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1474 = tensor.empty() : tensor<1x32x80x128xf32>
    %1475 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1473 : tensor<1x80x32x128xf32>) outs(%1474 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1476 = "tosa.reshape"(%1467) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1477 = tensor.empty() : tensor<1x32x80x128xf32>
    %1478 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1476 : tensor<1x80x32x128xf32>) outs(%1477 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1479 = "tosa.reshape"(%1472) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1480 = tensor.empty() : tensor<1x32x80x128xf32>
    %1481 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1479 : tensor<1x80x32x128xf32>) outs(%1480 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_931 = tensor.extract_slice %expanded_560[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_932 = tensor.extract_slice %extracted_slice_931[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_933 = tensor.extract_slice %extracted_slice_932[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_934 = tensor.extract_slice %expanded_562[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_935 = tensor.extract_slice %extracted_slice_934[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_936 = tensor.extract_slice %extracted_slice_935[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %1482 = tensor.empty() : tensor<1x80x128xf32>
    %1483 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_933 : tensor<1x1x80x128xf32>) outs(%1482 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1484 = tensor.empty() : tensor<80x128xf32>
    %1485 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1483 : tensor<1x80x128xf32>) outs(%1484 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1486 = tensor.empty() : tensor<1x80x128xf32>
    %1487 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_936 : tensor<1x1x80x128xf32>) outs(%1486 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1488 = tensor.empty() : tensor<80x128xf32>
    %1489 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1487 : tensor<1x80x128xf32>) outs(%1488 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1490 = tensor.empty() : tensor<1x80x128xf32>
    %1491 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1490 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1485[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1492 = "tosa.reshape"(%1491) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1493 = tensor.empty() : tensor<1x80x128xf32>
    %1494 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1493 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1489[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1495 = "tosa.reshape"(%1494) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1496 = "tosa.mul"(%1475, %1492) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_937 = tensor.extract_slice %1475[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_938 = tensor.extract_slice %1475[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1497 = tensor.empty() : tensor<1x32x80x64xf32>
    %1498 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_938 : tensor<1x32x80x64xf32>) outs(%1497 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1499 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_939 = tensor.insert_slice %1498 into %1499[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_940 = tensor.insert_slice %extracted_slice_937 into %inserted_slice_939[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1500 = "tosa.mul"(%inserted_slice_940, %1495) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1501 = "tosa.add"(%1496, %1500) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1502 = "tosa.mul"(%1478, %1492) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_941 = tensor.extract_slice %1478[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_942 = tensor.extract_slice %1478[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1503 = tensor.empty() : tensor<1x32x80x64xf32>
    %1504 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_942 : tensor<1x32x80x64xf32>) outs(%1503 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1505 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_943 = tensor.insert_slice %1504 into %1505[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_944 = tensor.insert_slice %extracted_slice_941 into %inserted_slice_943[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1506 = "tosa.mul"(%inserted_slice_944, %1495) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1507 = "tosa.add"(%1502, %1506) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1508 = tensor.empty() : tensor<1x32x128x80xf32>
    %1509 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1507 : tensor<1x32x80x128xf32>) outs(%1508 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %1510 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1511 = "tosa.add"(%1501, %1510) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1512 = "tosa.reshape"(%1511) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1513 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %1514 = "tosa.add"(%1509, %1513) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %1515 = "tosa.reshape"(%1514) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %1516 = "tosa.matmul"(%1512, %1515) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %1517 = "tosa.reshape"(%1516) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1518 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1519 = "tosa.reciprocal"(%1518) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1520 = "tosa.mul"(%1517, %1519) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1521 = "tosa.add"(%1520, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1522 = tensor.empty() : tensor<1x32x80x1xf32>
    %1523 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1522 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1524 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1521 : tensor<1x32x80x80xf32>) outs(%1522 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1525 = tensor.empty() : tensor<1x32x80x80xf32>
    %1526 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1521, %1524 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1525 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %1527 = tensor.empty() : tensor<1x32x80x1xf32>
    %1528 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1527 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1529 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1526 : tensor<1x32x80x80xf32>) outs(%1528 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1530 = tensor.empty() : tensor<1x32x80x80xf32>
    %1531 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1526, %1529 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1530 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %1532 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1533 = "tosa.add"(%1531, %1532) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1534 = "tosa.reshape"(%1533) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %1535 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1536 = "tosa.add"(%1481, %1535) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1537 = "tosa.reshape"(%1536) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1538 = "tosa.matmul"(%1534, %1537) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %1539 = "tosa.reshape"(%1538) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1540 = tensor.empty() : tensor<1x80x32x128xf32>
    %1541 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1539 : tensor<1x32x80x128xf32>) outs(%1540 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %1542 = "tosa.identity"(%1541) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %1543 = "tosa.reshape"(%1542) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %1544 = tensor.empty() : tensor<4096x4096xf32>
    %1545 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_226 : tensor<4096x4096xf32>) outs(%1544 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1546 = "tosa.reshape"(%1543) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_945 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1547 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1546, %1545 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_945 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1548 = "tosa.reshape"(%1547) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1549 = "tosa.add"(%1448, %1548) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1550 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_946 = arith.constant 2 : i32
    %1551 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1549 : tensor<1x80x4096xf32>) outs(%1550 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_946 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_947 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1552 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1551 : tensor<1x80x4096xf32>) outs(%cst_947 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1553 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1554 = "tosa.add"(%1552, %1553) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1555 = "tosa.rsqrt"(%1554) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1556 = "tosa.mul"(%1549, %1555) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1557 = "tosa.reshape"(%extracted_slice_22) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1558 = "tosa.mul"(%1557, %1556) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1559 = tensor.empty() : tensor<4096x11008xf32>
    %1560 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_228 : tensor<11008x4096xf32>) outs(%1559 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1561 = "tosa.reshape"(%1558) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_948 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1562 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1561, %1560 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_948 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1563 = "tosa.reshape"(%1562) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1564 = tensor.empty() : tensor<1x80x11008xf32>
    %1565 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1563 : tensor<1x80x11008xf32>) outs(%1564 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %1566 = tensor.empty() : tensor<4096x11008xf32>
    %1567 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_230 : tensor<11008x4096xf32>) outs(%1566 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1568 = "tosa.reshape"(%1558) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_949 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1569 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1568, %1567 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_949 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1570 = "tosa.reshape"(%1569) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1571 = "tosa.mul"(%1565, %1570) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %1572 = tensor.empty() : tensor<11008x4096xf32>
    %1573 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_232 : tensor<4096x11008xf32>) outs(%1572 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %1574 = "tosa.reshape"(%1571) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_950 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1575 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1574, %1573 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_950 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1576 = "tosa.reshape"(%1575) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1577 = "tosa.add"(%1549, %1576) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1578 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_951 = arith.constant 2 : i32
    %1579 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1577 : tensor<1x80x4096xf32>) outs(%1578 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_951 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_952 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1580 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1579 : tensor<1x80x4096xf32>) outs(%cst_952 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1581 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1582 = "tosa.add"(%1580, %1581) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1583 = "tosa.rsqrt"(%1582) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1584 = "tosa.mul"(%1577, %1583) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1585 = "tosa.reshape"(%extracted_slice_23) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1586 = "tosa.mul"(%1585, %1584) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1587 = tensor.empty() : tensor<4096x4096xf32>
    %1588 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_234 : tensor<4096x4096xf32>) outs(%1587 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1589 = "tosa.reshape"(%1586) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_953 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1590 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1589, %1588 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_953 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1591 = "tosa.reshape"(%1590) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1592 = tensor.empty() : tensor<4096x4096xf32>
    %1593 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_236 : tensor<4096x4096xf32>) outs(%1592 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1594 = "tosa.reshape"(%1586) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_954 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1595 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1594, %1593 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_954 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1596 = "tosa.reshape"(%1595) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1597 = tensor.empty() : tensor<4096x4096xf32>
    %1598 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_238 : tensor<4096x4096xf32>) outs(%1597 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1599 = "tosa.reshape"(%1586) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_955 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1600 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1599, %1598 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_955 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1601 = "tosa.reshape"(%1600) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1602 = "tosa.reshape"(%1591) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1603 = tensor.empty() : tensor<1x32x80x128xf32>
    %1604 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1602 : tensor<1x80x32x128xf32>) outs(%1603 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1605 = "tosa.reshape"(%1596) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1606 = tensor.empty() : tensor<1x32x80x128xf32>
    %1607 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1605 : tensor<1x80x32x128xf32>) outs(%1606 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1608 = "tosa.reshape"(%1601) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1609 = tensor.empty() : tensor<1x32x80x128xf32>
    %1610 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1608 : tensor<1x80x32x128xf32>) outs(%1609 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_956 = tensor.extract_slice %expanded_564[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_957 = tensor.extract_slice %extracted_slice_956[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_958 = tensor.extract_slice %extracted_slice_957[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_959 = tensor.extract_slice %expanded_566[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_960 = tensor.extract_slice %extracted_slice_959[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_961 = tensor.extract_slice %extracted_slice_960[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %1611 = tensor.empty() : tensor<1x80x128xf32>
    %1612 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_958 : tensor<1x1x80x128xf32>) outs(%1611 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1613 = tensor.empty() : tensor<80x128xf32>
    %1614 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1612 : tensor<1x80x128xf32>) outs(%1613 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1615 = tensor.empty() : tensor<1x80x128xf32>
    %1616 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_961 : tensor<1x1x80x128xf32>) outs(%1615 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1617 = tensor.empty() : tensor<80x128xf32>
    %1618 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1616 : tensor<1x80x128xf32>) outs(%1617 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1619 = tensor.empty() : tensor<1x80x128xf32>
    %1620 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1619 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1614[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1621 = "tosa.reshape"(%1620) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1622 = tensor.empty() : tensor<1x80x128xf32>
    %1623 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1622 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1618[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1624 = "tosa.reshape"(%1623) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1625 = "tosa.mul"(%1604, %1621) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_962 = tensor.extract_slice %1604[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_963 = tensor.extract_slice %1604[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1626 = tensor.empty() : tensor<1x32x80x64xf32>
    %1627 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_963 : tensor<1x32x80x64xf32>) outs(%1626 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1628 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_964 = tensor.insert_slice %1627 into %1628[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_965 = tensor.insert_slice %extracted_slice_962 into %inserted_slice_964[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1629 = "tosa.mul"(%inserted_slice_965, %1624) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1630 = "tosa.add"(%1625, %1629) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1631 = "tosa.mul"(%1607, %1621) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_966 = tensor.extract_slice %1607[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_967 = tensor.extract_slice %1607[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1632 = tensor.empty() : tensor<1x32x80x64xf32>
    %1633 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_967 : tensor<1x32x80x64xf32>) outs(%1632 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1634 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_968 = tensor.insert_slice %1633 into %1634[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_969 = tensor.insert_slice %extracted_slice_966 into %inserted_slice_968[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1635 = "tosa.mul"(%inserted_slice_969, %1624) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1636 = "tosa.add"(%1631, %1635) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1637 = tensor.empty() : tensor<1x32x128x80xf32>
    %1638 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1636 : tensor<1x32x80x128xf32>) outs(%1637 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %1639 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1640 = "tosa.add"(%1630, %1639) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1641 = "tosa.reshape"(%1640) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1642 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %1643 = "tosa.add"(%1638, %1642) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %1644 = "tosa.reshape"(%1643) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %1645 = "tosa.matmul"(%1641, %1644) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %1646 = "tosa.reshape"(%1645) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1647 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1648 = "tosa.reciprocal"(%1647) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1649 = "tosa.mul"(%1646, %1648) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1650 = "tosa.add"(%1649, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1651 = tensor.empty() : tensor<1x32x80x1xf32>
    %1652 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1651 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1653 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1650 : tensor<1x32x80x80xf32>) outs(%1651 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1654 = tensor.empty() : tensor<1x32x80x80xf32>
    %1655 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1650, %1653 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1654 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %1656 = tensor.empty() : tensor<1x32x80x1xf32>
    %1657 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1656 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1658 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1655 : tensor<1x32x80x80xf32>) outs(%1657 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1659 = tensor.empty() : tensor<1x32x80x80xf32>
    %1660 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1655, %1658 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1659 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %1661 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1662 = "tosa.add"(%1660, %1661) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1663 = "tosa.reshape"(%1662) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %1664 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1665 = "tosa.add"(%1610, %1664) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1666 = "tosa.reshape"(%1665) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1667 = "tosa.matmul"(%1663, %1666) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %1668 = "tosa.reshape"(%1667) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1669 = tensor.empty() : tensor<1x80x32x128xf32>
    %1670 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1668 : tensor<1x32x80x128xf32>) outs(%1669 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %1671 = "tosa.identity"(%1670) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %1672 = "tosa.reshape"(%1671) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %1673 = tensor.empty() : tensor<4096x4096xf32>
    %1674 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_240 : tensor<4096x4096xf32>) outs(%1673 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1675 = "tosa.reshape"(%1672) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_970 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1676 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1675, %1674 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_970 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1677 = "tosa.reshape"(%1676) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1678 = "tosa.add"(%1577, %1677) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1679 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_971 = arith.constant 2 : i32
    %1680 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1678 : tensor<1x80x4096xf32>) outs(%1679 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_971 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_972 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1681 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1680 : tensor<1x80x4096xf32>) outs(%cst_972 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1682 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1683 = "tosa.add"(%1681, %1682) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1684 = "tosa.rsqrt"(%1683) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1685 = "tosa.mul"(%1678, %1684) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1686 = "tosa.reshape"(%extracted_slice_24) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1687 = "tosa.mul"(%1686, %1685) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1688 = tensor.empty() : tensor<4096x11008xf32>
    %1689 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_242 : tensor<11008x4096xf32>) outs(%1688 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1690 = "tosa.reshape"(%1687) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_973 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1691 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1690, %1689 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_973 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1692 = "tosa.reshape"(%1691) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1693 = tensor.empty() : tensor<1x80x11008xf32>
    %1694 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1692 : tensor<1x80x11008xf32>) outs(%1693 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %1695 = tensor.empty() : tensor<4096x11008xf32>
    %1696 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_244 : tensor<11008x4096xf32>) outs(%1695 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1697 = "tosa.reshape"(%1687) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_974 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1698 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1697, %1696 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_974 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1699 = "tosa.reshape"(%1698) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1700 = "tosa.mul"(%1694, %1699) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %1701 = tensor.empty() : tensor<11008x4096xf32>
    %1702 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_246 : tensor<4096x11008xf32>) outs(%1701 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %1703 = "tosa.reshape"(%1700) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_975 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1704 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1703, %1702 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_975 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1705 = "tosa.reshape"(%1704) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1706 = "tosa.add"(%1678, %1705) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1707 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_976 = arith.constant 2 : i32
    %1708 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1706 : tensor<1x80x4096xf32>) outs(%1707 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_976 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_977 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1709 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1708 : tensor<1x80x4096xf32>) outs(%cst_977 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1710 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1711 = "tosa.add"(%1709, %1710) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1712 = "tosa.rsqrt"(%1711) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1713 = "tosa.mul"(%1706, %1712) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1714 = "tosa.reshape"(%extracted_slice_25) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1715 = "tosa.mul"(%1714, %1713) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1716 = tensor.empty() : tensor<4096x4096xf32>
    %1717 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_248 : tensor<4096x4096xf32>) outs(%1716 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1718 = "tosa.reshape"(%1715) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_978 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1719 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1718, %1717 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_978 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1720 = "tosa.reshape"(%1719) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1721 = tensor.empty() : tensor<4096x4096xf32>
    %1722 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_250 : tensor<4096x4096xf32>) outs(%1721 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1723 = "tosa.reshape"(%1715) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_979 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1724 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1723, %1722 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_979 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1725 = "tosa.reshape"(%1724) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1726 = tensor.empty() : tensor<4096x4096xf32>
    %1727 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_252 : tensor<4096x4096xf32>) outs(%1726 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1728 = "tosa.reshape"(%1715) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_980 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1729 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1728, %1727 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_980 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1730 = "tosa.reshape"(%1729) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1731 = "tosa.reshape"(%1720) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1732 = tensor.empty() : tensor<1x32x80x128xf32>
    %1733 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1731 : tensor<1x80x32x128xf32>) outs(%1732 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1734 = "tosa.reshape"(%1725) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1735 = tensor.empty() : tensor<1x32x80x128xf32>
    %1736 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1734 : tensor<1x80x32x128xf32>) outs(%1735 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1737 = "tosa.reshape"(%1730) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1738 = tensor.empty() : tensor<1x32x80x128xf32>
    %1739 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1737 : tensor<1x80x32x128xf32>) outs(%1738 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_981 = tensor.extract_slice %expanded_568[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_982 = tensor.extract_slice %extracted_slice_981[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_983 = tensor.extract_slice %extracted_slice_982[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_984 = tensor.extract_slice %expanded_570[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_985 = tensor.extract_slice %extracted_slice_984[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_986 = tensor.extract_slice %extracted_slice_985[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %1740 = tensor.empty() : tensor<1x80x128xf32>
    %1741 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_983 : tensor<1x1x80x128xf32>) outs(%1740 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1742 = tensor.empty() : tensor<80x128xf32>
    %1743 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1741 : tensor<1x80x128xf32>) outs(%1742 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1744 = tensor.empty() : tensor<1x80x128xf32>
    %1745 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_986 : tensor<1x1x80x128xf32>) outs(%1744 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1746 = tensor.empty() : tensor<80x128xf32>
    %1747 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1745 : tensor<1x80x128xf32>) outs(%1746 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1748 = tensor.empty() : tensor<1x80x128xf32>
    %1749 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1748 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1743[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1750 = "tosa.reshape"(%1749) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1751 = tensor.empty() : tensor<1x80x128xf32>
    %1752 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1751 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1747[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1753 = "tosa.reshape"(%1752) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1754 = "tosa.mul"(%1733, %1750) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_987 = tensor.extract_slice %1733[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_988 = tensor.extract_slice %1733[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1755 = tensor.empty() : tensor<1x32x80x64xf32>
    %1756 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_988 : tensor<1x32x80x64xf32>) outs(%1755 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1757 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_989 = tensor.insert_slice %1756 into %1757[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_990 = tensor.insert_slice %extracted_slice_987 into %inserted_slice_989[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1758 = "tosa.mul"(%inserted_slice_990, %1753) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1759 = "tosa.add"(%1754, %1758) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1760 = "tosa.mul"(%1736, %1750) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_991 = tensor.extract_slice %1736[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_992 = tensor.extract_slice %1736[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1761 = tensor.empty() : tensor<1x32x80x64xf32>
    %1762 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_992 : tensor<1x32x80x64xf32>) outs(%1761 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1763 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_993 = tensor.insert_slice %1762 into %1763[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_994 = tensor.insert_slice %extracted_slice_991 into %inserted_slice_993[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1764 = "tosa.mul"(%inserted_slice_994, %1753) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1765 = "tosa.add"(%1760, %1764) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1766 = tensor.empty() : tensor<1x32x128x80xf32>
    %1767 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1765 : tensor<1x32x80x128xf32>) outs(%1766 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %1768 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1769 = "tosa.add"(%1759, %1768) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1770 = "tosa.reshape"(%1769) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1771 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %1772 = "tosa.add"(%1767, %1771) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %1773 = "tosa.reshape"(%1772) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %1774 = "tosa.matmul"(%1770, %1773) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %1775 = "tosa.reshape"(%1774) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1776 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1777 = "tosa.reciprocal"(%1776) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1778 = "tosa.mul"(%1775, %1777) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1779 = "tosa.add"(%1778, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1780 = tensor.empty() : tensor<1x32x80x1xf32>
    %1781 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1780 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1782 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1779 : tensor<1x32x80x80xf32>) outs(%1780 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1783 = tensor.empty() : tensor<1x32x80x80xf32>
    %1784 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1779, %1782 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1783 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %1785 = tensor.empty() : tensor<1x32x80x1xf32>
    %1786 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1785 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1787 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1784 : tensor<1x32x80x80xf32>) outs(%1786 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1788 = tensor.empty() : tensor<1x32x80x80xf32>
    %1789 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1784, %1787 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1788 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %1790 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1791 = "tosa.add"(%1789, %1790) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1792 = "tosa.reshape"(%1791) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %1793 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1794 = "tosa.add"(%1739, %1793) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1795 = "tosa.reshape"(%1794) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1796 = "tosa.matmul"(%1792, %1795) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %1797 = "tosa.reshape"(%1796) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1798 = tensor.empty() : tensor<1x80x32x128xf32>
    %1799 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1797 : tensor<1x32x80x128xf32>) outs(%1798 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %1800 = "tosa.identity"(%1799) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %1801 = "tosa.reshape"(%1800) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %1802 = tensor.empty() : tensor<4096x4096xf32>
    %1803 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_254 : tensor<4096x4096xf32>) outs(%1802 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1804 = "tosa.reshape"(%1801) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_995 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1805 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1804, %1803 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_995 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1806 = "tosa.reshape"(%1805) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1807 = "tosa.add"(%1706, %1806) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1808 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_996 = arith.constant 2 : i32
    %1809 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1807 : tensor<1x80x4096xf32>) outs(%1808 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_996 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_997 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1810 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1809 : tensor<1x80x4096xf32>) outs(%cst_997 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1811 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1812 = "tosa.add"(%1810, %1811) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1813 = "tosa.rsqrt"(%1812) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1814 = "tosa.mul"(%1807, %1813) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1815 = "tosa.reshape"(%extracted_slice_26) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1816 = "tosa.mul"(%1815, %1814) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1817 = tensor.empty() : tensor<4096x11008xf32>
    %1818 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_256 : tensor<11008x4096xf32>) outs(%1817 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1819 = "tosa.reshape"(%1816) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_998 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1820 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1819, %1818 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_998 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1821 = "tosa.reshape"(%1820) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1822 = tensor.empty() : tensor<1x80x11008xf32>
    %1823 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1821 : tensor<1x80x11008xf32>) outs(%1822 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %1824 = tensor.empty() : tensor<4096x11008xf32>
    %1825 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_258 : tensor<11008x4096xf32>) outs(%1824 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1826 = "tosa.reshape"(%1816) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_999 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1827 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1826, %1825 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_999 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1828 = "tosa.reshape"(%1827) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1829 = "tosa.mul"(%1823, %1828) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %1830 = tensor.empty() : tensor<11008x4096xf32>
    %1831 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_260 : tensor<4096x11008xf32>) outs(%1830 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %1832 = "tosa.reshape"(%1829) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1000 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1833 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1832, %1831 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1000 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1834 = "tosa.reshape"(%1833) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1835 = "tosa.add"(%1807, %1834) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1836 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1001 = arith.constant 2 : i32
    %1837 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1835 : tensor<1x80x4096xf32>) outs(%1836 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1001 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1002 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1838 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1837 : tensor<1x80x4096xf32>) outs(%cst_1002 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1839 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1840 = "tosa.add"(%1838, %1839) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1841 = "tosa.rsqrt"(%1840) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1842 = "tosa.mul"(%1835, %1841) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1843 = "tosa.reshape"(%extracted_slice_27) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1844 = "tosa.mul"(%1843, %1842) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1845 = tensor.empty() : tensor<4096x4096xf32>
    %1846 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_262 : tensor<4096x4096xf32>) outs(%1845 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1847 = "tosa.reshape"(%1844) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1003 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1848 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1847, %1846 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1003 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1849 = "tosa.reshape"(%1848) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1850 = tensor.empty() : tensor<4096x4096xf32>
    %1851 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_264 : tensor<4096x4096xf32>) outs(%1850 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1852 = "tosa.reshape"(%1844) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1004 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1853 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1852, %1851 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1004 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1854 = "tosa.reshape"(%1853) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1855 = tensor.empty() : tensor<4096x4096xf32>
    %1856 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_266 : tensor<4096x4096xf32>) outs(%1855 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1857 = "tosa.reshape"(%1844) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1005 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1858 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1857, %1856 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1005 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1859 = "tosa.reshape"(%1858) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1860 = "tosa.reshape"(%1849) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1861 = tensor.empty() : tensor<1x32x80x128xf32>
    %1862 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1860 : tensor<1x80x32x128xf32>) outs(%1861 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1863 = "tosa.reshape"(%1854) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1864 = tensor.empty() : tensor<1x32x80x128xf32>
    %1865 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1863 : tensor<1x80x32x128xf32>) outs(%1864 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1866 = "tosa.reshape"(%1859) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1867 = tensor.empty() : tensor<1x32x80x128xf32>
    %1868 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1866 : tensor<1x80x32x128xf32>) outs(%1867 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1006 = tensor.extract_slice %expanded_572[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1007 = tensor.extract_slice %extracted_slice_1006[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1008 = tensor.extract_slice %extracted_slice_1007[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1009 = tensor.extract_slice %expanded_574[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1010 = tensor.extract_slice %extracted_slice_1009[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1011 = tensor.extract_slice %extracted_slice_1010[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %1869 = tensor.empty() : tensor<1x80x128xf32>
    %1870 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1008 : tensor<1x1x80x128xf32>) outs(%1869 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1871 = tensor.empty() : tensor<80x128xf32>
    %1872 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1870 : tensor<1x80x128xf32>) outs(%1871 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1873 = tensor.empty() : tensor<1x80x128xf32>
    %1874 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1011 : tensor<1x1x80x128xf32>) outs(%1873 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %1875 = tensor.empty() : tensor<80x128xf32>
    %1876 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1874 : tensor<1x80x128xf32>) outs(%1875 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %1877 = tensor.empty() : tensor<1x80x128xf32>
    %1878 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1877 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1872[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1879 = "tosa.reshape"(%1878) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1880 = tensor.empty() : tensor<1x80x128xf32>
    %1881 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%1880 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %1876[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %1882 = "tosa.reshape"(%1881) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %1883 = "tosa.mul"(%1862, %1879) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1012 = tensor.extract_slice %1862[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1013 = tensor.extract_slice %1862[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1884 = tensor.empty() : tensor<1x32x80x64xf32>
    %1885 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1013 : tensor<1x32x80x64xf32>) outs(%1884 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1886 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1014 = tensor.insert_slice %1885 into %1886[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1015 = tensor.insert_slice %extracted_slice_1012 into %inserted_slice_1014[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1887 = "tosa.mul"(%inserted_slice_1015, %1882) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1888 = "tosa.add"(%1883, %1887) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1889 = "tosa.mul"(%1865, %1879) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1016 = tensor.extract_slice %1865[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1017 = tensor.extract_slice %1865[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %1890 = tensor.empty() : tensor<1x32x80x64xf32>
    %1891 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1017 : tensor<1x32x80x64xf32>) outs(%1890 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %1892 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1018 = tensor.insert_slice %1891 into %1892[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1019 = tensor.insert_slice %extracted_slice_1016 into %inserted_slice_1018[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %1893 = "tosa.mul"(%inserted_slice_1019, %1882) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1894 = "tosa.add"(%1889, %1893) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1895 = tensor.empty() : tensor<1x32x128x80xf32>
    %1896 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1894 : tensor<1x32x80x128xf32>) outs(%1895 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %1897 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1898 = "tosa.add"(%1888, %1897) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1899 = "tosa.reshape"(%1898) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1900 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %1901 = "tosa.add"(%1896, %1900) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %1902 = "tosa.reshape"(%1901) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %1903 = "tosa.matmul"(%1899, %1902) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %1904 = "tosa.reshape"(%1903) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1905 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1906 = "tosa.reciprocal"(%1905) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1907 = "tosa.mul"(%1904, %1906) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1908 = "tosa.add"(%1907, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1909 = tensor.empty() : tensor<1x32x80x1xf32>
    %1910 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1909 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1911 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1908 : tensor<1x32x80x80xf32>) outs(%1909 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1912 = tensor.empty() : tensor<1x32x80x80xf32>
    %1913 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1908, %1911 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1912 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %1914 = tensor.empty() : tensor<1x32x80x1xf32>
    %1915 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1914 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %1916 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1913 : tensor<1x32x80x80xf32>) outs(%1915 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %1917 = tensor.empty() : tensor<1x32x80x80xf32>
    %1918 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1913, %1916 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%1917 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %1919 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %1920 = "tosa.add"(%1918, %1919) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %1921 = "tosa.reshape"(%1920) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %1922 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %1923 = "tosa.add"(%1868, %1922) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1924 = "tosa.reshape"(%1923) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %1925 = "tosa.matmul"(%1921, %1924) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %1926 = "tosa.reshape"(%1925) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %1927 = tensor.empty() : tensor<1x80x32x128xf32>
    %1928 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1926 : tensor<1x32x80x128xf32>) outs(%1927 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %1929 = "tosa.identity"(%1928) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %1930 = "tosa.reshape"(%1929) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %1931 = tensor.empty() : tensor<4096x4096xf32>
    %1932 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_268 : tensor<4096x4096xf32>) outs(%1931 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1933 = "tosa.reshape"(%1930) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1020 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1934 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1933, %1932 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1020 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1935 = "tosa.reshape"(%1934) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1936 = "tosa.add"(%1835, %1935) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1937 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1021 = arith.constant 2 : i32
    %1938 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1936 : tensor<1x80x4096xf32>) outs(%1937 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1021 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1022 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1939 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1938 : tensor<1x80x4096xf32>) outs(%cst_1022 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1940 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1941 = "tosa.add"(%1939, %1940) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1942 = "tosa.rsqrt"(%1941) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1943 = "tosa.mul"(%1936, %1942) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1944 = "tosa.reshape"(%extracted_slice_28) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1945 = "tosa.mul"(%1944, %1943) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1946 = tensor.empty() : tensor<4096x11008xf32>
    %1947 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_270 : tensor<11008x4096xf32>) outs(%1946 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1948 = "tosa.reshape"(%1945) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1023 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1949 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1948, %1947 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1023 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1950 = "tosa.reshape"(%1949) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1951 = tensor.empty() : tensor<1x80x11008xf32>
    %1952 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1950 : tensor<1x80x11008xf32>) outs(%1951 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %1953 = tensor.empty() : tensor<4096x11008xf32>
    %1954 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_272 : tensor<11008x4096xf32>) outs(%1953 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %1955 = "tosa.reshape"(%1945) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1024 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %1956 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1955, %1954 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1024 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %1957 = "tosa.reshape"(%1956) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %1958 = "tosa.mul"(%1952, %1957) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %1959 = tensor.empty() : tensor<11008x4096xf32>
    %1960 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_274 : tensor<4096x11008xf32>) outs(%1959 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %1961 = "tosa.reshape"(%1958) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1025 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1962 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1961, %1960 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1025 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1963 = "tosa.reshape"(%1962) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1964 = "tosa.add"(%1936, %1963) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1965 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1026 = arith.constant 2 : i32
    %1966 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1964 : tensor<1x80x4096xf32>) outs(%1965 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1026 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1027 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %1967 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%1966 : tensor<1x80x4096xf32>) outs(%cst_1027 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %1968 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %1969 = "tosa.add"(%1967, %1968) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1970 = "tosa.rsqrt"(%1969) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %1971 = "tosa.mul"(%1964, %1970) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %1972 = "tosa.reshape"(%extracted_slice_29) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %1973 = "tosa.mul"(%1972, %1971) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %1974 = tensor.empty() : tensor<4096x4096xf32>
    %1975 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_276 : tensor<4096x4096xf32>) outs(%1974 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1976 = "tosa.reshape"(%1973) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1028 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1977 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1976, %1975 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1028 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1978 = "tosa.reshape"(%1977) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1979 = tensor.empty() : tensor<4096x4096xf32>
    %1980 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_278 : tensor<4096x4096xf32>) outs(%1979 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1981 = "tosa.reshape"(%1973) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1029 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1982 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1981, %1980 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1029 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1983 = "tosa.reshape"(%1982) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1984 = tensor.empty() : tensor<4096x4096xf32>
    %1985 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_280 : tensor<4096x4096xf32>) outs(%1984 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %1986 = "tosa.reshape"(%1973) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1030 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %1987 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%1986, %1985 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1030 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %1988 = "tosa.reshape"(%1987) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %1989 = "tosa.reshape"(%1978) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1990 = tensor.empty() : tensor<1x32x80x128xf32>
    %1991 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1989 : tensor<1x80x32x128xf32>) outs(%1990 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1992 = "tosa.reshape"(%1983) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1993 = tensor.empty() : tensor<1x32x80x128xf32>
    %1994 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1992 : tensor<1x80x32x128xf32>) outs(%1993 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %1995 = "tosa.reshape"(%1988) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %1996 = tensor.empty() : tensor<1x32x80x128xf32>
    %1997 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1995 : tensor<1x80x32x128xf32>) outs(%1996 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1031 = tensor.extract_slice %expanded_576[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1032 = tensor.extract_slice %extracted_slice_1031[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1033 = tensor.extract_slice %extracted_slice_1032[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1034 = tensor.extract_slice %expanded_578[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1035 = tensor.extract_slice %extracted_slice_1034[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1036 = tensor.extract_slice %extracted_slice_1035[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %1998 = tensor.empty() : tensor<1x80x128xf32>
    %1999 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1033 : tensor<1x1x80x128xf32>) outs(%1998 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2000 = tensor.empty() : tensor<80x128xf32>
    %2001 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%1999 : tensor<1x80x128xf32>) outs(%2000 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2002 = tensor.empty() : tensor<1x80x128xf32>
    %2003 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1036 : tensor<1x1x80x128xf32>) outs(%2002 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2004 = tensor.empty() : tensor<80x128xf32>
    %2005 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2003 : tensor<1x80x128xf32>) outs(%2004 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2006 = tensor.empty() : tensor<1x80x128xf32>
    %2007 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2006 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2001[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2008 = "tosa.reshape"(%2007) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2009 = tensor.empty() : tensor<1x80x128xf32>
    %2010 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2009 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2005[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2011 = "tosa.reshape"(%2010) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2012 = "tosa.mul"(%1991, %2008) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1037 = tensor.extract_slice %1991[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1038 = tensor.extract_slice %1991[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2013 = tensor.empty() : tensor<1x32x80x64xf32>
    %2014 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1038 : tensor<1x32x80x64xf32>) outs(%2013 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2015 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1039 = tensor.insert_slice %2014 into %2015[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1040 = tensor.insert_slice %extracted_slice_1037 into %inserted_slice_1039[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2016 = "tosa.mul"(%inserted_slice_1040, %2011) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2017 = "tosa.add"(%2012, %2016) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2018 = "tosa.mul"(%1994, %2008) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1041 = tensor.extract_slice %1994[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1042 = tensor.extract_slice %1994[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2019 = tensor.empty() : tensor<1x32x80x64xf32>
    %2020 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1042 : tensor<1x32x80x64xf32>) outs(%2019 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2021 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1043 = tensor.insert_slice %2020 into %2021[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1044 = tensor.insert_slice %extracted_slice_1041 into %inserted_slice_1043[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2022 = "tosa.mul"(%inserted_slice_1044, %2011) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2023 = "tosa.add"(%2018, %2022) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2024 = tensor.empty() : tensor<1x32x128x80xf32>
    %2025 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2023 : tensor<1x32x80x128xf32>) outs(%2024 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %2026 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2027 = "tosa.add"(%2017, %2026) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2028 = "tosa.reshape"(%2027) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2029 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %2030 = "tosa.add"(%2025, %2029) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %2031 = "tosa.reshape"(%2030) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %2032 = "tosa.matmul"(%2028, %2031) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %2033 = "tosa.reshape"(%2032) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2034 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2035 = "tosa.reciprocal"(%2034) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2036 = "tosa.mul"(%2033, %2035) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2037 = "tosa.add"(%2036, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2038 = tensor.empty() : tensor<1x32x80x1xf32>
    %2039 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2038 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2040 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2037 : tensor<1x32x80x80xf32>) outs(%2038 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2041 = tensor.empty() : tensor<1x32x80x80xf32>
    %2042 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2037, %2040 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2041 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %2043 = tensor.empty() : tensor<1x32x80x1xf32>
    %2044 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2043 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2045 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2042 : tensor<1x32x80x80xf32>) outs(%2044 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2046 = tensor.empty() : tensor<1x32x80x80xf32>
    %2047 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2042, %2045 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2046 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %2048 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2049 = "tosa.add"(%2047, %2048) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2050 = "tosa.reshape"(%2049) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %2051 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2052 = "tosa.add"(%1997, %2051) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2053 = "tosa.reshape"(%2052) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2054 = "tosa.matmul"(%2050, %2053) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %2055 = "tosa.reshape"(%2054) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2056 = tensor.empty() : tensor<1x80x32x128xf32>
    %2057 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2055 : tensor<1x32x80x128xf32>) outs(%2056 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %2058 = "tosa.identity"(%2057) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %2059 = "tosa.reshape"(%2058) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %2060 = tensor.empty() : tensor<4096x4096xf32>
    %2061 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_282 : tensor<4096x4096xf32>) outs(%2060 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2062 = "tosa.reshape"(%2059) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1045 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2063 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2062, %2061 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1045 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2064 = "tosa.reshape"(%2063) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2065 = "tosa.add"(%1964, %2064) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2066 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1046 = arith.constant 2 : i32
    %2067 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2065 : tensor<1x80x4096xf32>) outs(%2066 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1046 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1047 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2068 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2067 : tensor<1x80x4096xf32>) outs(%cst_1047 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2069 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2070 = "tosa.add"(%2068, %2069) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2071 = "tosa.rsqrt"(%2070) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2072 = "tosa.mul"(%2065, %2071) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2073 = "tosa.reshape"(%extracted_slice_30) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2074 = "tosa.mul"(%2073, %2072) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2075 = tensor.empty() : tensor<4096x11008xf32>
    %2076 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_284 : tensor<11008x4096xf32>) outs(%2075 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2077 = "tosa.reshape"(%2074) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1048 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2078 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2077, %2076 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1048 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2079 = "tosa.reshape"(%2078) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2080 = tensor.empty() : tensor<1x80x11008xf32>
    %2081 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2079 : tensor<1x80x11008xf32>) outs(%2080 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %2082 = tensor.empty() : tensor<4096x11008xf32>
    %2083 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_286 : tensor<11008x4096xf32>) outs(%2082 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2084 = "tosa.reshape"(%2074) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1049 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2085 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2084, %2083 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1049 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2086 = "tosa.reshape"(%2085) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2087 = "tosa.mul"(%2081, %2086) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %2088 = tensor.empty() : tensor<11008x4096xf32>
    %2089 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_288 : tensor<4096x11008xf32>) outs(%2088 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %2090 = "tosa.reshape"(%2087) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1050 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2091 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2090, %2089 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1050 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2092 = "tosa.reshape"(%2091) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2093 = "tosa.add"(%2065, %2092) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2094 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1051 = arith.constant 2 : i32
    %2095 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2093 : tensor<1x80x4096xf32>) outs(%2094 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1051 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1052 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2096 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2095 : tensor<1x80x4096xf32>) outs(%cst_1052 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2097 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2098 = "tosa.add"(%2096, %2097) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2099 = "tosa.rsqrt"(%2098) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2100 = "tosa.mul"(%2093, %2099) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2101 = "tosa.reshape"(%extracted_slice_31) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2102 = "tosa.mul"(%2101, %2100) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2103 = tensor.empty() : tensor<4096x4096xf32>
    %2104 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_290 : tensor<4096x4096xf32>) outs(%2103 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2105 = "tosa.reshape"(%2102) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1053 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2106 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2105, %2104 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1053 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2107 = "tosa.reshape"(%2106) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2108 = tensor.empty() : tensor<4096x4096xf32>
    %2109 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_292 : tensor<4096x4096xf32>) outs(%2108 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2110 = "tosa.reshape"(%2102) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1054 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2111 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2110, %2109 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1054 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2112 = "tosa.reshape"(%2111) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2113 = tensor.empty() : tensor<4096x4096xf32>
    %2114 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_294 : tensor<4096x4096xf32>) outs(%2113 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2115 = "tosa.reshape"(%2102) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1055 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2116 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2115, %2114 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1055 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2117 = "tosa.reshape"(%2116) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2118 = "tosa.reshape"(%2107) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2119 = tensor.empty() : tensor<1x32x80x128xf32>
    %2120 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2118 : tensor<1x80x32x128xf32>) outs(%2119 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2121 = "tosa.reshape"(%2112) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2122 = tensor.empty() : tensor<1x32x80x128xf32>
    %2123 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2121 : tensor<1x80x32x128xf32>) outs(%2122 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2124 = "tosa.reshape"(%2117) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2125 = tensor.empty() : tensor<1x32x80x128xf32>
    %2126 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2124 : tensor<1x80x32x128xf32>) outs(%2125 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1056 = tensor.extract_slice %expanded_580[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1057 = tensor.extract_slice %extracted_slice_1056[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1058 = tensor.extract_slice %extracted_slice_1057[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1059 = tensor.extract_slice %expanded_582[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1060 = tensor.extract_slice %extracted_slice_1059[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1061 = tensor.extract_slice %extracted_slice_1060[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %2127 = tensor.empty() : tensor<1x80x128xf32>
    %2128 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1058 : tensor<1x1x80x128xf32>) outs(%2127 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2129 = tensor.empty() : tensor<80x128xf32>
    %2130 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2128 : tensor<1x80x128xf32>) outs(%2129 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2131 = tensor.empty() : tensor<1x80x128xf32>
    %2132 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1061 : tensor<1x1x80x128xf32>) outs(%2131 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2133 = tensor.empty() : tensor<80x128xf32>
    %2134 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2132 : tensor<1x80x128xf32>) outs(%2133 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2135 = tensor.empty() : tensor<1x80x128xf32>
    %2136 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2135 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2130[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2137 = "tosa.reshape"(%2136) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2138 = tensor.empty() : tensor<1x80x128xf32>
    %2139 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2138 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2134[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2140 = "tosa.reshape"(%2139) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2141 = "tosa.mul"(%2120, %2137) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1062 = tensor.extract_slice %2120[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1063 = tensor.extract_slice %2120[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2142 = tensor.empty() : tensor<1x32x80x64xf32>
    %2143 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1063 : tensor<1x32x80x64xf32>) outs(%2142 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2144 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1064 = tensor.insert_slice %2143 into %2144[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1065 = tensor.insert_slice %extracted_slice_1062 into %inserted_slice_1064[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2145 = "tosa.mul"(%inserted_slice_1065, %2140) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2146 = "tosa.add"(%2141, %2145) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2147 = "tosa.mul"(%2123, %2137) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1066 = tensor.extract_slice %2123[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1067 = tensor.extract_slice %2123[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2148 = tensor.empty() : tensor<1x32x80x64xf32>
    %2149 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1067 : tensor<1x32x80x64xf32>) outs(%2148 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2150 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1068 = tensor.insert_slice %2149 into %2150[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1069 = tensor.insert_slice %extracted_slice_1066 into %inserted_slice_1068[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2151 = "tosa.mul"(%inserted_slice_1069, %2140) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2152 = "tosa.add"(%2147, %2151) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2153 = tensor.empty() : tensor<1x32x128x80xf32>
    %2154 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2152 : tensor<1x32x80x128xf32>) outs(%2153 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %2155 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2156 = "tosa.add"(%2146, %2155) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2157 = "tosa.reshape"(%2156) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2158 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %2159 = "tosa.add"(%2154, %2158) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %2160 = "tosa.reshape"(%2159) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %2161 = "tosa.matmul"(%2157, %2160) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %2162 = "tosa.reshape"(%2161) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2163 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2164 = "tosa.reciprocal"(%2163) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2165 = "tosa.mul"(%2162, %2164) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2166 = "tosa.add"(%2165, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2167 = tensor.empty() : tensor<1x32x80x1xf32>
    %2168 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2167 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2169 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2166 : tensor<1x32x80x80xf32>) outs(%2167 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2170 = tensor.empty() : tensor<1x32x80x80xf32>
    %2171 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2166, %2169 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2170 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %2172 = tensor.empty() : tensor<1x32x80x1xf32>
    %2173 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2172 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2174 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2171 : tensor<1x32x80x80xf32>) outs(%2173 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2175 = tensor.empty() : tensor<1x32x80x80xf32>
    %2176 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2171, %2174 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2175 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %2177 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2178 = "tosa.add"(%2176, %2177) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2179 = "tosa.reshape"(%2178) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %2180 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2181 = "tosa.add"(%2126, %2180) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2182 = "tosa.reshape"(%2181) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2183 = "tosa.matmul"(%2179, %2182) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %2184 = "tosa.reshape"(%2183) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2185 = tensor.empty() : tensor<1x80x32x128xf32>
    %2186 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2184 : tensor<1x32x80x128xf32>) outs(%2185 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %2187 = "tosa.identity"(%2186) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %2188 = "tosa.reshape"(%2187) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %2189 = tensor.empty() : tensor<4096x4096xf32>
    %2190 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_296 : tensor<4096x4096xf32>) outs(%2189 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2191 = "tosa.reshape"(%2188) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1070 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2192 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2191, %2190 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1070 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2193 = "tosa.reshape"(%2192) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2194 = "tosa.add"(%2093, %2193) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2195 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1071 = arith.constant 2 : i32
    %2196 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2194 : tensor<1x80x4096xf32>) outs(%2195 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1071 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1072 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2197 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2196 : tensor<1x80x4096xf32>) outs(%cst_1072 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2198 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2199 = "tosa.add"(%2197, %2198) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2200 = "tosa.rsqrt"(%2199) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2201 = "tosa.mul"(%2194, %2200) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2202 = "tosa.reshape"(%extracted_slice_32) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2203 = "tosa.mul"(%2202, %2201) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2204 = tensor.empty() : tensor<4096x11008xf32>
    %2205 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_298 : tensor<11008x4096xf32>) outs(%2204 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2206 = "tosa.reshape"(%2203) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1073 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2207 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2206, %2205 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1073 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2208 = "tosa.reshape"(%2207) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2209 = tensor.empty() : tensor<1x80x11008xf32>
    %2210 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2208 : tensor<1x80x11008xf32>) outs(%2209 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %2211 = tensor.empty() : tensor<4096x11008xf32>
    %2212 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_300 : tensor<11008x4096xf32>) outs(%2211 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2213 = "tosa.reshape"(%2203) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1074 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2214 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2213, %2212 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1074 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2215 = "tosa.reshape"(%2214) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2216 = "tosa.mul"(%2210, %2215) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %2217 = tensor.empty() : tensor<11008x4096xf32>
    %2218 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_302 : tensor<4096x11008xf32>) outs(%2217 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %2219 = "tosa.reshape"(%2216) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1075 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2220 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2219, %2218 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1075 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2221 = "tosa.reshape"(%2220) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2222 = "tosa.add"(%2194, %2221) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2223 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1076 = arith.constant 2 : i32
    %2224 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2222 : tensor<1x80x4096xf32>) outs(%2223 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1076 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1077 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2225 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2224 : tensor<1x80x4096xf32>) outs(%cst_1077 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2226 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2227 = "tosa.add"(%2225, %2226) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2228 = "tosa.rsqrt"(%2227) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2229 = "tosa.mul"(%2222, %2228) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2230 = "tosa.reshape"(%extracted_slice_33) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2231 = "tosa.mul"(%2230, %2229) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2232 = tensor.empty() : tensor<4096x4096xf32>
    %2233 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_304 : tensor<4096x4096xf32>) outs(%2232 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2234 = "tosa.reshape"(%2231) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1078 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2235 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2234, %2233 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1078 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2236 = "tosa.reshape"(%2235) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2237 = tensor.empty() : tensor<4096x4096xf32>
    %2238 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_306 : tensor<4096x4096xf32>) outs(%2237 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2239 = "tosa.reshape"(%2231) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1079 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2240 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2239, %2238 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1079 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2241 = "tosa.reshape"(%2240) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2242 = tensor.empty() : tensor<4096x4096xf32>
    %2243 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_308 : tensor<4096x4096xf32>) outs(%2242 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2244 = "tosa.reshape"(%2231) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1080 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2245 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2244, %2243 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1080 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2246 = "tosa.reshape"(%2245) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2247 = "tosa.reshape"(%2236) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2248 = tensor.empty() : tensor<1x32x80x128xf32>
    %2249 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2247 : tensor<1x80x32x128xf32>) outs(%2248 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2250 = "tosa.reshape"(%2241) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2251 = tensor.empty() : tensor<1x32x80x128xf32>
    %2252 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2250 : tensor<1x80x32x128xf32>) outs(%2251 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2253 = "tosa.reshape"(%2246) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2254 = tensor.empty() : tensor<1x32x80x128xf32>
    %2255 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2253 : tensor<1x80x32x128xf32>) outs(%2254 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1081 = tensor.extract_slice %expanded_584[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1082 = tensor.extract_slice %extracted_slice_1081[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1083 = tensor.extract_slice %extracted_slice_1082[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1084 = tensor.extract_slice %expanded_586[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1085 = tensor.extract_slice %extracted_slice_1084[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1086 = tensor.extract_slice %extracted_slice_1085[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %2256 = tensor.empty() : tensor<1x80x128xf32>
    %2257 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1083 : tensor<1x1x80x128xf32>) outs(%2256 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2258 = tensor.empty() : tensor<80x128xf32>
    %2259 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2257 : tensor<1x80x128xf32>) outs(%2258 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2260 = tensor.empty() : tensor<1x80x128xf32>
    %2261 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1086 : tensor<1x1x80x128xf32>) outs(%2260 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2262 = tensor.empty() : tensor<80x128xf32>
    %2263 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2261 : tensor<1x80x128xf32>) outs(%2262 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2264 = tensor.empty() : tensor<1x80x128xf32>
    %2265 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2264 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2259[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2266 = "tosa.reshape"(%2265) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2267 = tensor.empty() : tensor<1x80x128xf32>
    %2268 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2267 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2263[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2269 = "tosa.reshape"(%2268) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2270 = "tosa.mul"(%2249, %2266) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1087 = tensor.extract_slice %2249[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1088 = tensor.extract_slice %2249[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2271 = tensor.empty() : tensor<1x32x80x64xf32>
    %2272 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1088 : tensor<1x32x80x64xf32>) outs(%2271 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2273 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1089 = tensor.insert_slice %2272 into %2273[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1090 = tensor.insert_slice %extracted_slice_1087 into %inserted_slice_1089[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2274 = "tosa.mul"(%inserted_slice_1090, %2269) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2275 = "tosa.add"(%2270, %2274) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2276 = "tosa.mul"(%2252, %2266) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1091 = tensor.extract_slice %2252[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1092 = tensor.extract_slice %2252[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2277 = tensor.empty() : tensor<1x32x80x64xf32>
    %2278 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1092 : tensor<1x32x80x64xf32>) outs(%2277 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2279 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1093 = tensor.insert_slice %2278 into %2279[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1094 = tensor.insert_slice %extracted_slice_1091 into %inserted_slice_1093[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2280 = "tosa.mul"(%inserted_slice_1094, %2269) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2281 = "tosa.add"(%2276, %2280) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2282 = tensor.empty() : tensor<1x32x128x80xf32>
    %2283 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2281 : tensor<1x32x80x128xf32>) outs(%2282 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %2284 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2285 = "tosa.add"(%2275, %2284) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2286 = "tosa.reshape"(%2285) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2287 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %2288 = "tosa.add"(%2283, %2287) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %2289 = "tosa.reshape"(%2288) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %2290 = "tosa.matmul"(%2286, %2289) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %2291 = "tosa.reshape"(%2290) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2292 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2293 = "tosa.reciprocal"(%2292) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2294 = "tosa.mul"(%2291, %2293) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2295 = "tosa.add"(%2294, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2296 = tensor.empty() : tensor<1x32x80x1xf32>
    %2297 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2296 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2298 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2295 : tensor<1x32x80x80xf32>) outs(%2296 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2299 = tensor.empty() : tensor<1x32x80x80xf32>
    %2300 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2295, %2298 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2299 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %2301 = tensor.empty() : tensor<1x32x80x1xf32>
    %2302 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2301 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2303 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2300 : tensor<1x32x80x80xf32>) outs(%2302 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2304 = tensor.empty() : tensor<1x32x80x80xf32>
    %2305 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2300, %2303 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2304 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %2306 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2307 = "tosa.add"(%2305, %2306) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2308 = "tosa.reshape"(%2307) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %2309 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2310 = "tosa.add"(%2255, %2309) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2311 = "tosa.reshape"(%2310) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2312 = "tosa.matmul"(%2308, %2311) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %2313 = "tosa.reshape"(%2312) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2314 = tensor.empty() : tensor<1x80x32x128xf32>
    %2315 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2313 : tensor<1x32x80x128xf32>) outs(%2314 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %2316 = "tosa.identity"(%2315) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %2317 = "tosa.reshape"(%2316) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %2318 = tensor.empty() : tensor<4096x4096xf32>
    %2319 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_310 : tensor<4096x4096xf32>) outs(%2318 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2320 = "tosa.reshape"(%2317) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1095 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2321 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2320, %2319 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1095 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2322 = "tosa.reshape"(%2321) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2323 = "tosa.add"(%2222, %2322) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2324 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1096 = arith.constant 2 : i32
    %2325 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2323 : tensor<1x80x4096xf32>) outs(%2324 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1096 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1097 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2326 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2325 : tensor<1x80x4096xf32>) outs(%cst_1097 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2327 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2328 = "tosa.add"(%2326, %2327) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2329 = "tosa.rsqrt"(%2328) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2330 = "tosa.mul"(%2323, %2329) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2331 = "tosa.reshape"(%extracted_slice_34) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2332 = "tosa.mul"(%2331, %2330) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2333 = tensor.empty() : tensor<4096x11008xf32>
    %2334 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_312 : tensor<11008x4096xf32>) outs(%2333 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2335 = "tosa.reshape"(%2332) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1098 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2336 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2335, %2334 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1098 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2337 = "tosa.reshape"(%2336) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2338 = tensor.empty() : tensor<1x80x11008xf32>
    %2339 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2337 : tensor<1x80x11008xf32>) outs(%2338 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %2340 = tensor.empty() : tensor<4096x11008xf32>
    %2341 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_314 : tensor<11008x4096xf32>) outs(%2340 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2342 = "tosa.reshape"(%2332) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1099 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2343 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2342, %2341 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1099 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2344 = "tosa.reshape"(%2343) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2345 = "tosa.mul"(%2339, %2344) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %2346 = tensor.empty() : tensor<11008x4096xf32>
    %2347 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_316 : tensor<4096x11008xf32>) outs(%2346 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %2348 = "tosa.reshape"(%2345) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1100 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2349 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2348, %2347 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1100 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2350 = "tosa.reshape"(%2349) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2351 = "tosa.add"(%2323, %2350) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2352 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1101 = arith.constant 2 : i32
    %2353 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2351 : tensor<1x80x4096xf32>) outs(%2352 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1101 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1102 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2354 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2353 : tensor<1x80x4096xf32>) outs(%cst_1102 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2355 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2356 = "tosa.add"(%2354, %2355) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2357 = "tosa.rsqrt"(%2356) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2358 = "tosa.mul"(%2351, %2357) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2359 = "tosa.reshape"(%extracted_slice_35) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2360 = "tosa.mul"(%2359, %2358) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2361 = tensor.empty() : tensor<4096x4096xf32>
    %2362 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_318 : tensor<4096x4096xf32>) outs(%2361 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2363 = "tosa.reshape"(%2360) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1103 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2364 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2363, %2362 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1103 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2365 = "tosa.reshape"(%2364) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2366 = tensor.empty() : tensor<4096x4096xf32>
    %2367 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_320 : tensor<4096x4096xf32>) outs(%2366 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2368 = "tosa.reshape"(%2360) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1104 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2369 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2368, %2367 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1104 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2370 = "tosa.reshape"(%2369) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2371 = tensor.empty() : tensor<4096x4096xf32>
    %2372 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_322 : tensor<4096x4096xf32>) outs(%2371 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2373 = "tosa.reshape"(%2360) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1105 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2374 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2373, %2372 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1105 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2375 = "tosa.reshape"(%2374) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2376 = "tosa.reshape"(%2365) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2377 = tensor.empty() : tensor<1x32x80x128xf32>
    %2378 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2376 : tensor<1x80x32x128xf32>) outs(%2377 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2379 = "tosa.reshape"(%2370) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2380 = tensor.empty() : tensor<1x32x80x128xf32>
    %2381 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2379 : tensor<1x80x32x128xf32>) outs(%2380 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2382 = "tosa.reshape"(%2375) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2383 = tensor.empty() : tensor<1x32x80x128xf32>
    %2384 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2382 : tensor<1x80x32x128xf32>) outs(%2383 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1106 = tensor.extract_slice %expanded_588[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1107 = tensor.extract_slice %extracted_slice_1106[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1108 = tensor.extract_slice %extracted_slice_1107[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1109 = tensor.extract_slice %expanded_590[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1110 = tensor.extract_slice %extracted_slice_1109[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1111 = tensor.extract_slice %extracted_slice_1110[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %2385 = tensor.empty() : tensor<1x80x128xf32>
    %2386 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1108 : tensor<1x1x80x128xf32>) outs(%2385 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2387 = tensor.empty() : tensor<80x128xf32>
    %2388 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2386 : tensor<1x80x128xf32>) outs(%2387 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2389 = tensor.empty() : tensor<1x80x128xf32>
    %2390 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1111 : tensor<1x1x80x128xf32>) outs(%2389 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2391 = tensor.empty() : tensor<80x128xf32>
    %2392 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2390 : tensor<1x80x128xf32>) outs(%2391 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2393 = tensor.empty() : tensor<1x80x128xf32>
    %2394 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2393 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2388[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2395 = "tosa.reshape"(%2394) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2396 = tensor.empty() : tensor<1x80x128xf32>
    %2397 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2396 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2392[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2398 = "tosa.reshape"(%2397) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2399 = "tosa.mul"(%2378, %2395) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1112 = tensor.extract_slice %2378[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1113 = tensor.extract_slice %2378[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2400 = tensor.empty() : tensor<1x32x80x64xf32>
    %2401 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1113 : tensor<1x32x80x64xf32>) outs(%2400 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2402 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1114 = tensor.insert_slice %2401 into %2402[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1115 = tensor.insert_slice %extracted_slice_1112 into %inserted_slice_1114[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2403 = "tosa.mul"(%inserted_slice_1115, %2398) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2404 = "tosa.add"(%2399, %2403) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2405 = "tosa.mul"(%2381, %2395) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1116 = tensor.extract_slice %2381[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1117 = tensor.extract_slice %2381[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2406 = tensor.empty() : tensor<1x32x80x64xf32>
    %2407 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1117 : tensor<1x32x80x64xf32>) outs(%2406 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2408 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1118 = tensor.insert_slice %2407 into %2408[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1119 = tensor.insert_slice %extracted_slice_1116 into %inserted_slice_1118[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2409 = "tosa.mul"(%inserted_slice_1119, %2398) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2410 = "tosa.add"(%2405, %2409) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2411 = tensor.empty() : tensor<1x32x128x80xf32>
    %2412 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2410 : tensor<1x32x80x128xf32>) outs(%2411 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %2413 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2414 = "tosa.add"(%2404, %2413) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2415 = "tosa.reshape"(%2414) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2416 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %2417 = "tosa.add"(%2412, %2416) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %2418 = "tosa.reshape"(%2417) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %2419 = "tosa.matmul"(%2415, %2418) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %2420 = "tosa.reshape"(%2419) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2421 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2422 = "tosa.reciprocal"(%2421) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2423 = "tosa.mul"(%2420, %2422) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2424 = "tosa.add"(%2423, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2425 = tensor.empty() : tensor<1x32x80x1xf32>
    %2426 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2425 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2427 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2424 : tensor<1x32x80x80xf32>) outs(%2425 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2428 = tensor.empty() : tensor<1x32x80x80xf32>
    %2429 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2424, %2427 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2428 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %2430 = tensor.empty() : tensor<1x32x80x1xf32>
    %2431 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2430 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2432 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2429 : tensor<1x32x80x80xf32>) outs(%2431 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2433 = tensor.empty() : tensor<1x32x80x80xf32>
    %2434 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2429, %2432 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2433 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %2435 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2436 = "tosa.add"(%2434, %2435) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2437 = "tosa.reshape"(%2436) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %2438 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2439 = "tosa.add"(%2384, %2438) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2440 = "tosa.reshape"(%2439) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2441 = "tosa.matmul"(%2437, %2440) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %2442 = "tosa.reshape"(%2441) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2443 = tensor.empty() : tensor<1x80x32x128xf32>
    %2444 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2442 : tensor<1x32x80x128xf32>) outs(%2443 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %2445 = "tosa.identity"(%2444) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %2446 = "tosa.reshape"(%2445) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %2447 = tensor.empty() : tensor<4096x4096xf32>
    %2448 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_324 : tensor<4096x4096xf32>) outs(%2447 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2449 = "tosa.reshape"(%2446) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1120 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2450 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2449, %2448 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1120 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2451 = "tosa.reshape"(%2450) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2452 = "tosa.add"(%2351, %2451) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2453 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1121 = arith.constant 2 : i32
    %2454 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2452 : tensor<1x80x4096xf32>) outs(%2453 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1121 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1122 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2455 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2454 : tensor<1x80x4096xf32>) outs(%cst_1122 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2456 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2457 = "tosa.add"(%2455, %2456) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2458 = "tosa.rsqrt"(%2457) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2459 = "tosa.mul"(%2452, %2458) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2460 = "tosa.reshape"(%extracted_slice_36) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2461 = "tosa.mul"(%2460, %2459) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2462 = tensor.empty() : tensor<4096x11008xf32>
    %2463 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_326 : tensor<11008x4096xf32>) outs(%2462 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2464 = "tosa.reshape"(%2461) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1123 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2465 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2464, %2463 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1123 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2466 = "tosa.reshape"(%2465) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2467 = tensor.empty() : tensor<1x80x11008xf32>
    %2468 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2466 : tensor<1x80x11008xf32>) outs(%2467 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %2469 = tensor.empty() : tensor<4096x11008xf32>
    %2470 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_328 : tensor<11008x4096xf32>) outs(%2469 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2471 = "tosa.reshape"(%2461) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1124 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2472 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2471, %2470 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1124 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2473 = "tosa.reshape"(%2472) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2474 = "tosa.mul"(%2468, %2473) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %2475 = tensor.empty() : tensor<11008x4096xf32>
    %2476 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_330 : tensor<4096x11008xf32>) outs(%2475 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %2477 = "tosa.reshape"(%2474) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1125 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2478 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2477, %2476 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1125 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2479 = "tosa.reshape"(%2478) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2480 = "tosa.add"(%2452, %2479) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2481 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1126 = arith.constant 2 : i32
    %2482 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2480 : tensor<1x80x4096xf32>) outs(%2481 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1126 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1127 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2483 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2482 : tensor<1x80x4096xf32>) outs(%cst_1127 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2484 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2485 = "tosa.add"(%2483, %2484) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2486 = "tosa.rsqrt"(%2485) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2487 = "tosa.mul"(%2480, %2486) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2488 = "tosa.reshape"(%extracted_slice_37) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2489 = "tosa.mul"(%2488, %2487) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2490 = tensor.empty() : tensor<4096x4096xf32>
    %2491 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_332 : tensor<4096x4096xf32>) outs(%2490 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2492 = "tosa.reshape"(%2489) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1128 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2493 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2492, %2491 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1128 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2494 = "tosa.reshape"(%2493) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2495 = tensor.empty() : tensor<4096x4096xf32>
    %2496 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_334 : tensor<4096x4096xf32>) outs(%2495 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2497 = "tosa.reshape"(%2489) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1129 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2498 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2497, %2496 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1129 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2499 = "tosa.reshape"(%2498) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2500 = tensor.empty() : tensor<4096x4096xf32>
    %2501 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_336 : tensor<4096x4096xf32>) outs(%2500 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2502 = "tosa.reshape"(%2489) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1130 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2503 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2502, %2501 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1130 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2504 = "tosa.reshape"(%2503) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2505 = "tosa.reshape"(%2494) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2506 = tensor.empty() : tensor<1x32x80x128xf32>
    %2507 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2505 : tensor<1x80x32x128xf32>) outs(%2506 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2508 = "tosa.reshape"(%2499) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2509 = tensor.empty() : tensor<1x32x80x128xf32>
    %2510 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2508 : tensor<1x80x32x128xf32>) outs(%2509 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2511 = "tosa.reshape"(%2504) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2512 = tensor.empty() : tensor<1x32x80x128xf32>
    %2513 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2511 : tensor<1x80x32x128xf32>) outs(%2512 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1131 = tensor.extract_slice %expanded_592[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1132 = tensor.extract_slice %extracted_slice_1131[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1133 = tensor.extract_slice %extracted_slice_1132[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1134 = tensor.extract_slice %expanded_594[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1135 = tensor.extract_slice %extracted_slice_1134[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1136 = tensor.extract_slice %extracted_slice_1135[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %2514 = tensor.empty() : tensor<1x80x128xf32>
    %2515 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1133 : tensor<1x1x80x128xf32>) outs(%2514 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2516 = tensor.empty() : tensor<80x128xf32>
    %2517 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2515 : tensor<1x80x128xf32>) outs(%2516 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2518 = tensor.empty() : tensor<1x80x128xf32>
    %2519 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1136 : tensor<1x1x80x128xf32>) outs(%2518 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2520 = tensor.empty() : tensor<80x128xf32>
    %2521 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2519 : tensor<1x80x128xf32>) outs(%2520 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2522 = tensor.empty() : tensor<1x80x128xf32>
    %2523 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2522 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2517[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2524 = "tosa.reshape"(%2523) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2525 = tensor.empty() : tensor<1x80x128xf32>
    %2526 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2525 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2521[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2527 = "tosa.reshape"(%2526) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2528 = "tosa.mul"(%2507, %2524) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1137 = tensor.extract_slice %2507[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1138 = tensor.extract_slice %2507[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2529 = tensor.empty() : tensor<1x32x80x64xf32>
    %2530 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1138 : tensor<1x32x80x64xf32>) outs(%2529 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2531 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1139 = tensor.insert_slice %2530 into %2531[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1140 = tensor.insert_slice %extracted_slice_1137 into %inserted_slice_1139[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2532 = "tosa.mul"(%inserted_slice_1140, %2527) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2533 = "tosa.add"(%2528, %2532) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2534 = "tosa.mul"(%2510, %2524) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1141 = tensor.extract_slice %2510[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1142 = tensor.extract_slice %2510[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2535 = tensor.empty() : tensor<1x32x80x64xf32>
    %2536 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1142 : tensor<1x32x80x64xf32>) outs(%2535 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2537 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1143 = tensor.insert_slice %2536 into %2537[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1144 = tensor.insert_slice %extracted_slice_1141 into %inserted_slice_1143[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2538 = "tosa.mul"(%inserted_slice_1144, %2527) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2539 = "tosa.add"(%2534, %2538) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2540 = tensor.empty() : tensor<1x32x128x80xf32>
    %2541 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2539 : tensor<1x32x80x128xf32>) outs(%2540 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %2542 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2543 = "tosa.add"(%2533, %2542) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2544 = "tosa.reshape"(%2543) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2545 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %2546 = "tosa.add"(%2541, %2545) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %2547 = "tosa.reshape"(%2546) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %2548 = "tosa.matmul"(%2544, %2547) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %2549 = "tosa.reshape"(%2548) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2550 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2551 = "tosa.reciprocal"(%2550) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2552 = "tosa.mul"(%2549, %2551) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2553 = "tosa.add"(%2552, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2554 = tensor.empty() : tensor<1x32x80x1xf32>
    %2555 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2554 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2556 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2553 : tensor<1x32x80x80xf32>) outs(%2554 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2557 = tensor.empty() : tensor<1x32x80x80xf32>
    %2558 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2553, %2556 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2557 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %2559 = tensor.empty() : tensor<1x32x80x1xf32>
    %2560 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2559 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2561 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2558 : tensor<1x32x80x80xf32>) outs(%2560 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2562 = tensor.empty() : tensor<1x32x80x80xf32>
    %2563 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2558, %2561 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2562 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %2564 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2565 = "tosa.add"(%2563, %2564) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2566 = "tosa.reshape"(%2565) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %2567 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2568 = "tosa.add"(%2513, %2567) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2569 = "tosa.reshape"(%2568) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2570 = "tosa.matmul"(%2566, %2569) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %2571 = "tosa.reshape"(%2570) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2572 = tensor.empty() : tensor<1x80x32x128xf32>
    %2573 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2571 : tensor<1x32x80x128xf32>) outs(%2572 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %2574 = "tosa.identity"(%2573) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %2575 = "tosa.reshape"(%2574) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %2576 = tensor.empty() : tensor<4096x4096xf32>
    %2577 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_338 : tensor<4096x4096xf32>) outs(%2576 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2578 = "tosa.reshape"(%2575) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1145 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2579 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2578, %2577 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1145 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2580 = "tosa.reshape"(%2579) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2581 = "tosa.add"(%2480, %2580) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2582 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1146 = arith.constant 2 : i32
    %2583 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2581 : tensor<1x80x4096xf32>) outs(%2582 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1146 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1147 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2584 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2583 : tensor<1x80x4096xf32>) outs(%cst_1147 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2585 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2586 = "tosa.add"(%2584, %2585) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2587 = "tosa.rsqrt"(%2586) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2588 = "tosa.mul"(%2581, %2587) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2589 = "tosa.reshape"(%extracted_slice_38) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2590 = "tosa.mul"(%2589, %2588) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2591 = tensor.empty() : tensor<4096x11008xf32>
    %2592 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_340 : tensor<11008x4096xf32>) outs(%2591 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2593 = "tosa.reshape"(%2590) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1148 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2594 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2593, %2592 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1148 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2595 = "tosa.reshape"(%2594) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2596 = tensor.empty() : tensor<1x80x11008xf32>
    %2597 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2595 : tensor<1x80x11008xf32>) outs(%2596 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %2598 = tensor.empty() : tensor<4096x11008xf32>
    %2599 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_342 : tensor<11008x4096xf32>) outs(%2598 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2600 = "tosa.reshape"(%2590) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1149 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2601 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2600, %2599 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1149 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2602 = "tosa.reshape"(%2601) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2603 = "tosa.mul"(%2597, %2602) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %2604 = tensor.empty() : tensor<11008x4096xf32>
    %2605 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_344 : tensor<4096x11008xf32>) outs(%2604 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %2606 = "tosa.reshape"(%2603) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1150 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2607 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2606, %2605 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1150 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2608 = "tosa.reshape"(%2607) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2609 = "tosa.add"(%2581, %2608) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2610 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1151 = arith.constant 2 : i32
    %2611 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2609 : tensor<1x80x4096xf32>) outs(%2610 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1151 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1152 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2612 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2611 : tensor<1x80x4096xf32>) outs(%cst_1152 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2613 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2614 = "tosa.add"(%2612, %2613) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2615 = "tosa.rsqrt"(%2614) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2616 = "tosa.mul"(%2609, %2615) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2617 = "tosa.reshape"(%extracted_slice_39) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2618 = "tosa.mul"(%2617, %2616) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2619 = tensor.empty() : tensor<4096x4096xf32>
    %2620 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_346 : tensor<4096x4096xf32>) outs(%2619 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2621 = "tosa.reshape"(%2618) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1153 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2622 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2621, %2620 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1153 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2623 = "tosa.reshape"(%2622) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2624 = tensor.empty() : tensor<4096x4096xf32>
    %2625 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_348 : tensor<4096x4096xf32>) outs(%2624 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2626 = "tosa.reshape"(%2618) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1154 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2627 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2626, %2625 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1154 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2628 = "tosa.reshape"(%2627) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2629 = tensor.empty() : tensor<4096x4096xf32>
    %2630 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_350 : tensor<4096x4096xf32>) outs(%2629 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2631 = "tosa.reshape"(%2618) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1155 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2632 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2631, %2630 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1155 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2633 = "tosa.reshape"(%2632) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2634 = "tosa.reshape"(%2623) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2635 = tensor.empty() : tensor<1x32x80x128xf32>
    %2636 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2634 : tensor<1x80x32x128xf32>) outs(%2635 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2637 = "tosa.reshape"(%2628) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2638 = tensor.empty() : tensor<1x32x80x128xf32>
    %2639 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2637 : tensor<1x80x32x128xf32>) outs(%2638 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2640 = "tosa.reshape"(%2633) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2641 = tensor.empty() : tensor<1x32x80x128xf32>
    %2642 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2640 : tensor<1x80x32x128xf32>) outs(%2641 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1156 = tensor.extract_slice %expanded_596[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1157 = tensor.extract_slice %extracted_slice_1156[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1158 = tensor.extract_slice %extracted_slice_1157[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1159 = tensor.extract_slice %expanded_598[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1160 = tensor.extract_slice %extracted_slice_1159[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1161 = tensor.extract_slice %extracted_slice_1160[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %2643 = tensor.empty() : tensor<1x80x128xf32>
    %2644 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1158 : tensor<1x1x80x128xf32>) outs(%2643 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2645 = tensor.empty() : tensor<80x128xf32>
    %2646 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2644 : tensor<1x80x128xf32>) outs(%2645 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2647 = tensor.empty() : tensor<1x80x128xf32>
    %2648 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1161 : tensor<1x1x80x128xf32>) outs(%2647 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2649 = tensor.empty() : tensor<80x128xf32>
    %2650 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2648 : tensor<1x80x128xf32>) outs(%2649 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2651 = tensor.empty() : tensor<1x80x128xf32>
    %2652 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2651 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2646[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2653 = "tosa.reshape"(%2652) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2654 = tensor.empty() : tensor<1x80x128xf32>
    %2655 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2654 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2650[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2656 = "tosa.reshape"(%2655) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2657 = "tosa.mul"(%2636, %2653) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1162 = tensor.extract_slice %2636[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1163 = tensor.extract_slice %2636[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2658 = tensor.empty() : tensor<1x32x80x64xf32>
    %2659 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1163 : tensor<1x32x80x64xf32>) outs(%2658 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2660 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1164 = tensor.insert_slice %2659 into %2660[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1165 = tensor.insert_slice %extracted_slice_1162 into %inserted_slice_1164[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2661 = "tosa.mul"(%inserted_slice_1165, %2656) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2662 = "tosa.add"(%2657, %2661) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2663 = "tosa.mul"(%2639, %2653) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1166 = tensor.extract_slice %2639[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1167 = tensor.extract_slice %2639[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2664 = tensor.empty() : tensor<1x32x80x64xf32>
    %2665 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1167 : tensor<1x32x80x64xf32>) outs(%2664 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2666 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1168 = tensor.insert_slice %2665 into %2666[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1169 = tensor.insert_slice %extracted_slice_1166 into %inserted_slice_1168[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2667 = "tosa.mul"(%inserted_slice_1169, %2656) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2668 = "tosa.add"(%2663, %2667) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2669 = tensor.empty() : tensor<1x32x128x80xf32>
    %2670 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2668 : tensor<1x32x80x128xf32>) outs(%2669 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %2671 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2672 = "tosa.add"(%2662, %2671) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2673 = "tosa.reshape"(%2672) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2674 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %2675 = "tosa.add"(%2670, %2674) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %2676 = "tosa.reshape"(%2675) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %2677 = "tosa.matmul"(%2673, %2676) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %2678 = "tosa.reshape"(%2677) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2679 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2680 = "tosa.reciprocal"(%2679) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2681 = "tosa.mul"(%2678, %2680) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2682 = "tosa.add"(%2681, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2683 = tensor.empty() : tensor<1x32x80x1xf32>
    %2684 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2683 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2685 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2682 : tensor<1x32x80x80xf32>) outs(%2683 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2686 = tensor.empty() : tensor<1x32x80x80xf32>
    %2687 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2682, %2685 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2686 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %2688 = tensor.empty() : tensor<1x32x80x1xf32>
    %2689 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2688 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2690 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2687 : tensor<1x32x80x80xf32>) outs(%2689 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2691 = tensor.empty() : tensor<1x32x80x80xf32>
    %2692 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2687, %2690 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2691 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %2693 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2694 = "tosa.add"(%2692, %2693) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2695 = "tosa.reshape"(%2694) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %2696 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2697 = "tosa.add"(%2642, %2696) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2698 = "tosa.reshape"(%2697) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2699 = "tosa.matmul"(%2695, %2698) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %2700 = "tosa.reshape"(%2699) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2701 = tensor.empty() : tensor<1x80x32x128xf32>
    %2702 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2700 : tensor<1x32x80x128xf32>) outs(%2701 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %2703 = "tosa.identity"(%2702) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %2704 = "tosa.reshape"(%2703) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %2705 = tensor.empty() : tensor<4096x4096xf32>
    %2706 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_352 : tensor<4096x4096xf32>) outs(%2705 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2707 = "tosa.reshape"(%2704) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1170 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2708 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2707, %2706 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1170 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2709 = "tosa.reshape"(%2708) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2710 = "tosa.add"(%2609, %2709) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2711 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1171 = arith.constant 2 : i32
    %2712 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2710 : tensor<1x80x4096xf32>) outs(%2711 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1171 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1172 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2713 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2712 : tensor<1x80x4096xf32>) outs(%cst_1172 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2714 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2715 = "tosa.add"(%2713, %2714) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2716 = "tosa.rsqrt"(%2715) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2717 = "tosa.mul"(%2710, %2716) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2718 = "tosa.reshape"(%extracted_slice_40) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2719 = "tosa.mul"(%2718, %2717) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2720 = tensor.empty() : tensor<4096x11008xf32>
    %2721 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_354 : tensor<11008x4096xf32>) outs(%2720 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2722 = "tosa.reshape"(%2719) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1173 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2723 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2722, %2721 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1173 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2724 = "tosa.reshape"(%2723) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2725 = tensor.empty() : tensor<1x80x11008xf32>
    %2726 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2724 : tensor<1x80x11008xf32>) outs(%2725 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %2727 = tensor.empty() : tensor<4096x11008xf32>
    %2728 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_356 : tensor<11008x4096xf32>) outs(%2727 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2729 = "tosa.reshape"(%2719) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1174 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2730 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2729, %2728 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1174 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2731 = "tosa.reshape"(%2730) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2732 = "tosa.mul"(%2726, %2731) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %2733 = tensor.empty() : tensor<11008x4096xf32>
    %2734 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_358 : tensor<4096x11008xf32>) outs(%2733 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %2735 = "tosa.reshape"(%2732) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1175 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2736 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2735, %2734 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1175 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2737 = "tosa.reshape"(%2736) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2738 = "tosa.add"(%2710, %2737) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2739 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1176 = arith.constant 2 : i32
    %2740 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2738 : tensor<1x80x4096xf32>) outs(%2739 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1176 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1177 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2741 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2740 : tensor<1x80x4096xf32>) outs(%cst_1177 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2742 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2743 = "tosa.add"(%2741, %2742) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2744 = "tosa.rsqrt"(%2743) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2745 = "tosa.mul"(%2738, %2744) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2746 = "tosa.reshape"(%extracted_slice_41) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2747 = "tosa.mul"(%2746, %2745) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2748 = tensor.empty() : tensor<4096x4096xf32>
    %2749 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_360 : tensor<4096x4096xf32>) outs(%2748 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2750 = "tosa.reshape"(%2747) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1178 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2751 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2750, %2749 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1178 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2752 = "tosa.reshape"(%2751) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2753 = tensor.empty() : tensor<4096x4096xf32>
    %2754 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_362 : tensor<4096x4096xf32>) outs(%2753 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2755 = "tosa.reshape"(%2747) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1179 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2756 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2755, %2754 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1179 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2757 = "tosa.reshape"(%2756) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2758 = tensor.empty() : tensor<4096x4096xf32>
    %2759 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_364 : tensor<4096x4096xf32>) outs(%2758 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2760 = "tosa.reshape"(%2747) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1180 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2761 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2760, %2759 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1180 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2762 = "tosa.reshape"(%2761) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2763 = "tosa.reshape"(%2752) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2764 = tensor.empty() : tensor<1x32x80x128xf32>
    %2765 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2763 : tensor<1x80x32x128xf32>) outs(%2764 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2766 = "tosa.reshape"(%2757) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2767 = tensor.empty() : tensor<1x32x80x128xf32>
    %2768 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2766 : tensor<1x80x32x128xf32>) outs(%2767 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2769 = "tosa.reshape"(%2762) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2770 = tensor.empty() : tensor<1x32x80x128xf32>
    %2771 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2769 : tensor<1x80x32x128xf32>) outs(%2770 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1181 = tensor.extract_slice %expanded_600[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1182 = tensor.extract_slice %extracted_slice_1181[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1183 = tensor.extract_slice %extracted_slice_1182[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1184 = tensor.extract_slice %expanded_602[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1185 = tensor.extract_slice %extracted_slice_1184[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1186 = tensor.extract_slice %extracted_slice_1185[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %2772 = tensor.empty() : tensor<1x80x128xf32>
    %2773 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1183 : tensor<1x1x80x128xf32>) outs(%2772 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2774 = tensor.empty() : tensor<80x128xf32>
    %2775 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2773 : tensor<1x80x128xf32>) outs(%2774 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2776 = tensor.empty() : tensor<1x80x128xf32>
    %2777 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1186 : tensor<1x1x80x128xf32>) outs(%2776 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2778 = tensor.empty() : tensor<80x128xf32>
    %2779 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2777 : tensor<1x80x128xf32>) outs(%2778 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2780 = tensor.empty() : tensor<1x80x128xf32>
    %2781 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2780 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2775[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2782 = "tosa.reshape"(%2781) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2783 = tensor.empty() : tensor<1x80x128xf32>
    %2784 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2783 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2779[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2785 = "tosa.reshape"(%2784) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2786 = "tosa.mul"(%2765, %2782) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1187 = tensor.extract_slice %2765[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1188 = tensor.extract_slice %2765[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2787 = tensor.empty() : tensor<1x32x80x64xf32>
    %2788 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1188 : tensor<1x32x80x64xf32>) outs(%2787 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2789 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1189 = tensor.insert_slice %2788 into %2789[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1190 = tensor.insert_slice %extracted_slice_1187 into %inserted_slice_1189[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2790 = "tosa.mul"(%inserted_slice_1190, %2785) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2791 = "tosa.add"(%2786, %2790) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2792 = "tosa.mul"(%2768, %2782) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1191 = tensor.extract_slice %2768[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1192 = tensor.extract_slice %2768[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2793 = tensor.empty() : tensor<1x32x80x64xf32>
    %2794 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1192 : tensor<1x32x80x64xf32>) outs(%2793 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2795 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1193 = tensor.insert_slice %2794 into %2795[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1194 = tensor.insert_slice %extracted_slice_1191 into %inserted_slice_1193[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2796 = "tosa.mul"(%inserted_slice_1194, %2785) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2797 = "tosa.add"(%2792, %2796) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2798 = tensor.empty() : tensor<1x32x128x80xf32>
    %2799 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2797 : tensor<1x32x80x128xf32>) outs(%2798 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %2800 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2801 = "tosa.add"(%2791, %2800) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2802 = "tosa.reshape"(%2801) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2803 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %2804 = "tosa.add"(%2799, %2803) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %2805 = "tosa.reshape"(%2804) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %2806 = "tosa.matmul"(%2802, %2805) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %2807 = "tosa.reshape"(%2806) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2808 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2809 = "tosa.reciprocal"(%2808) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2810 = "tosa.mul"(%2807, %2809) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2811 = "tosa.add"(%2810, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2812 = tensor.empty() : tensor<1x32x80x1xf32>
    %2813 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2812 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2814 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2811 : tensor<1x32x80x80xf32>) outs(%2812 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2815 = tensor.empty() : tensor<1x32x80x80xf32>
    %2816 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2811, %2814 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2815 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %2817 = tensor.empty() : tensor<1x32x80x1xf32>
    %2818 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2817 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2819 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2816 : tensor<1x32x80x80xf32>) outs(%2818 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2820 = tensor.empty() : tensor<1x32x80x80xf32>
    %2821 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2816, %2819 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2820 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %2822 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2823 = "tosa.add"(%2821, %2822) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2824 = "tosa.reshape"(%2823) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %2825 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2826 = "tosa.add"(%2771, %2825) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2827 = "tosa.reshape"(%2826) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2828 = "tosa.matmul"(%2824, %2827) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %2829 = "tosa.reshape"(%2828) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2830 = tensor.empty() : tensor<1x80x32x128xf32>
    %2831 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2829 : tensor<1x32x80x128xf32>) outs(%2830 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %2832 = "tosa.identity"(%2831) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %2833 = "tosa.reshape"(%2832) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %2834 = tensor.empty() : tensor<4096x4096xf32>
    %2835 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_366 : tensor<4096x4096xf32>) outs(%2834 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2836 = "tosa.reshape"(%2833) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1195 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2837 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2836, %2835 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1195 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2838 = "tosa.reshape"(%2837) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2839 = "tosa.add"(%2738, %2838) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2840 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1196 = arith.constant 2 : i32
    %2841 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2839 : tensor<1x80x4096xf32>) outs(%2840 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1196 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1197 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2842 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2841 : tensor<1x80x4096xf32>) outs(%cst_1197 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2843 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2844 = "tosa.add"(%2842, %2843) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2845 = "tosa.rsqrt"(%2844) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2846 = "tosa.mul"(%2839, %2845) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2847 = "tosa.reshape"(%extracted_slice_42) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2848 = "tosa.mul"(%2847, %2846) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2849 = tensor.empty() : tensor<4096x11008xf32>
    %2850 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_368 : tensor<11008x4096xf32>) outs(%2849 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2851 = "tosa.reshape"(%2848) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1198 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2852 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2851, %2850 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1198 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2853 = "tosa.reshape"(%2852) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2854 = tensor.empty() : tensor<1x80x11008xf32>
    %2855 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2853 : tensor<1x80x11008xf32>) outs(%2854 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %2856 = tensor.empty() : tensor<4096x11008xf32>
    %2857 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_370 : tensor<11008x4096xf32>) outs(%2856 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2858 = "tosa.reshape"(%2848) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1199 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2859 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2858, %2857 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1199 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2860 = "tosa.reshape"(%2859) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2861 = "tosa.mul"(%2855, %2860) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %2862 = tensor.empty() : tensor<11008x4096xf32>
    %2863 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_372 : tensor<4096x11008xf32>) outs(%2862 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %2864 = "tosa.reshape"(%2861) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1200 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2865 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2864, %2863 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1200 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2866 = "tosa.reshape"(%2865) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2867 = "tosa.add"(%2839, %2866) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2868 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1201 = arith.constant 2 : i32
    %2869 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2867 : tensor<1x80x4096xf32>) outs(%2868 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1201 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1202 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2870 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2869 : tensor<1x80x4096xf32>) outs(%cst_1202 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2871 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2872 = "tosa.add"(%2870, %2871) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2873 = "tosa.rsqrt"(%2872) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2874 = "tosa.mul"(%2867, %2873) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2875 = "tosa.reshape"(%extracted_slice_43) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2876 = "tosa.mul"(%2875, %2874) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2877 = tensor.empty() : tensor<4096x4096xf32>
    %2878 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_374 : tensor<4096x4096xf32>) outs(%2877 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2879 = "tosa.reshape"(%2876) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1203 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2880 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2879, %2878 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1203 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2881 = "tosa.reshape"(%2880) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2882 = tensor.empty() : tensor<4096x4096xf32>
    %2883 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_376 : tensor<4096x4096xf32>) outs(%2882 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2884 = "tosa.reshape"(%2876) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1204 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2885 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2884, %2883 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1204 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2886 = "tosa.reshape"(%2885) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2887 = tensor.empty() : tensor<4096x4096xf32>
    %2888 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_378 : tensor<4096x4096xf32>) outs(%2887 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2889 = "tosa.reshape"(%2876) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1205 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2890 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2889, %2888 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1205 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2891 = "tosa.reshape"(%2890) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2892 = "tosa.reshape"(%2881) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2893 = tensor.empty() : tensor<1x32x80x128xf32>
    %2894 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2892 : tensor<1x80x32x128xf32>) outs(%2893 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2895 = "tosa.reshape"(%2886) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2896 = tensor.empty() : tensor<1x32x80x128xf32>
    %2897 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2895 : tensor<1x80x32x128xf32>) outs(%2896 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %2898 = "tosa.reshape"(%2891) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %2899 = tensor.empty() : tensor<1x32x80x128xf32>
    %2900 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2898 : tensor<1x80x32x128xf32>) outs(%2899 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1206 = tensor.extract_slice %expanded_604[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1207 = tensor.extract_slice %extracted_slice_1206[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1208 = tensor.extract_slice %extracted_slice_1207[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1209 = tensor.extract_slice %expanded_606[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1210 = tensor.extract_slice %extracted_slice_1209[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1211 = tensor.extract_slice %extracted_slice_1210[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %2901 = tensor.empty() : tensor<1x80x128xf32>
    %2902 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1208 : tensor<1x1x80x128xf32>) outs(%2901 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2903 = tensor.empty() : tensor<80x128xf32>
    %2904 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2902 : tensor<1x80x128xf32>) outs(%2903 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2905 = tensor.empty() : tensor<1x80x128xf32>
    %2906 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1211 : tensor<1x1x80x128xf32>) outs(%2905 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %2907 = tensor.empty() : tensor<80x128xf32>
    %2908 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%2906 : tensor<1x80x128xf32>) outs(%2907 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %2909 = tensor.empty() : tensor<1x80x128xf32>
    %2910 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2909 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2904[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2911 = "tosa.reshape"(%2910) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2912 = tensor.empty() : tensor<1x80x128xf32>
    %2913 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%2912 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %2908[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %2914 = "tosa.reshape"(%2913) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %2915 = "tosa.mul"(%2894, %2911) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1212 = tensor.extract_slice %2894[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1213 = tensor.extract_slice %2894[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2916 = tensor.empty() : tensor<1x32x80x64xf32>
    %2917 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1213 : tensor<1x32x80x64xf32>) outs(%2916 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2918 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1214 = tensor.insert_slice %2917 into %2918[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1215 = tensor.insert_slice %extracted_slice_1212 into %inserted_slice_1214[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2919 = "tosa.mul"(%inserted_slice_1215, %2914) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2920 = "tosa.add"(%2915, %2919) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2921 = "tosa.mul"(%2897, %2911) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1216 = tensor.extract_slice %2897[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1217 = tensor.extract_slice %2897[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %2922 = tensor.empty() : tensor<1x32x80x64xf32>
    %2923 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1217 : tensor<1x32x80x64xf32>) outs(%2922 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %2924 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1218 = tensor.insert_slice %2923 into %2924[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1219 = tensor.insert_slice %extracted_slice_1216 into %inserted_slice_1218[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %2925 = "tosa.mul"(%inserted_slice_1219, %2914) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2926 = "tosa.add"(%2921, %2925) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2927 = tensor.empty() : tensor<1x32x128x80xf32>
    %2928 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2926 : tensor<1x32x80x128xf32>) outs(%2927 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %2929 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2930 = "tosa.add"(%2920, %2929) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2931 = "tosa.reshape"(%2930) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2932 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %2933 = "tosa.add"(%2928, %2932) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %2934 = "tosa.reshape"(%2933) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %2935 = "tosa.matmul"(%2931, %2934) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %2936 = "tosa.reshape"(%2935) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2937 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2938 = "tosa.reciprocal"(%2937) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2939 = "tosa.mul"(%2936, %2938) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2940 = "tosa.add"(%2939, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2941 = tensor.empty() : tensor<1x32x80x1xf32>
    %2942 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2941 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2943 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2940 : tensor<1x32x80x80xf32>) outs(%2941 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2944 = tensor.empty() : tensor<1x32x80x80xf32>
    %2945 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2940, %2943 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2944 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %2946 = tensor.empty() : tensor<1x32x80x1xf32>
    %2947 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%2946 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %2948 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2945 : tensor<1x32x80x80xf32>) outs(%2947 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %2949 = tensor.empty() : tensor<1x32x80x80xf32>
    %2950 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2945, %2948 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%2949 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %2951 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %2952 = "tosa.add"(%2950, %2951) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %2953 = "tosa.reshape"(%2952) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %2954 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %2955 = "tosa.add"(%2900, %2954) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2956 = "tosa.reshape"(%2955) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %2957 = "tosa.matmul"(%2953, %2956) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %2958 = "tosa.reshape"(%2957) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %2959 = tensor.empty() : tensor<1x80x32x128xf32>
    %2960 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2958 : tensor<1x32x80x128xf32>) outs(%2959 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %2961 = "tosa.identity"(%2960) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %2962 = "tosa.reshape"(%2961) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %2963 = tensor.empty() : tensor<4096x4096xf32>
    %2964 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_380 : tensor<4096x4096xf32>) outs(%2963 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %2965 = "tosa.reshape"(%2962) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1220 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2966 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2965, %2964 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1220 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2967 = "tosa.reshape"(%2966) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2968 = "tosa.add"(%2867, %2967) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2969 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1221 = arith.constant 2 : i32
    %2970 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2968 : tensor<1x80x4096xf32>) outs(%2969 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1221 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1222 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2971 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2970 : tensor<1x80x4096xf32>) outs(%cst_1222 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %2972 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %2973 = "tosa.add"(%2971, %2972) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2974 = "tosa.rsqrt"(%2973) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %2975 = "tosa.mul"(%2968, %2974) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %2976 = "tosa.reshape"(%extracted_slice_44) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %2977 = "tosa.mul"(%2976, %2975) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2978 = tensor.empty() : tensor<4096x11008xf32>
    %2979 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_382 : tensor<11008x4096xf32>) outs(%2978 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2980 = "tosa.reshape"(%2977) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1223 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2981 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2980, %2979 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1223 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2982 = "tosa.reshape"(%2981) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2983 = tensor.empty() : tensor<1x80x11008xf32>
    %2984 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2982 : tensor<1x80x11008xf32>) outs(%2983 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %2985 = tensor.empty() : tensor<4096x11008xf32>
    %2986 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_384 : tensor<11008x4096xf32>) outs(%2985 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %2987 = "tosa.reshape"(%2977) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1224 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %2988 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2987, %2986 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1224 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %2989 = "tosa.reshape"(%2988) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %2990 = "tosa.mul"(%2984, %2989) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %2991 = tensor.empty() : tensor<11008x4096xf32>
    %2992 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_386 : tensor<4096x11008xf32>) outs(%2991 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %2993 = "tosa.reshape"(%2990) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1225 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %2994 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%2993, %2992 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1225 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %2995 = "tosa.reshape"(%2994) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %2996 = "tosa.add"(%2968, %2995) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %2997 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1226 = arith.constant 2 : i32
    %2998 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2996 : tensor<1x80x4096xf32>) outs(%2997 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1226 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1227 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %2999 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%2998 : tensor<1x80x4096xf32>) outs(%cst_1227 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3000 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3001 = "tosa.add"(%2999, %3000) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3002 = "tosa.rsqrt"(%3001) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3003 = "tosa.mul"(%2996, %3002) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3004 = "tosa.reshape"(%extracted_slice_45) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3005 = "tosa.mul"(%3004, %3003) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3006 = tensor.empty() : tensor<4096x4096xf32>
    %3007 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_388 : tensor<4096x4096xf32>) outs(%3006 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3008 = "tosa.reshape"(%3005) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1228 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3009 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3008, %3007 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1228 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3010 = "tosa.reshape"(%3009) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3011 = tensor.empty() : tensor<4096x4096xf32>
    %3012 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_390 : tensor<4096x4096xf32>) outs(%3011 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3013 = "tosa.reshape"(%3005) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1229 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3014 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3013, %3012 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1229 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3015 = "tosa.reshape"(%3014) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3016 = tensor.empty() : tensor<4096x4096xf32>
    %3017 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_392 : tensor<4096x4096xf32>) outs(%3016 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3018 = "tosa.reshape"(%3005) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1230 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3019 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3018, %3017 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1230 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3020 = "tosa.reshape"(%3019) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3021 = "tosa.reshape"(%3010) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3022 = tensor.empty() : tensor<1x32x80x128xf32>
    %3023 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3021 : tensor<1x80x32x128xf32>) outs(%3022 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3024 = "tosa.reshape"(%3015) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3025 = tensor.empty() : tensor<1x32x80x128xf32>
    %3026 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3024 : tensor<1x80x32x128xf32>) outs(%3025 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3027 = "tosa.reshape"(%3020) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3028 = tensor.empty() : tensor<1x32x80x128xf32>
    %3029 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3027 : tensor<1x80x32x128xf32>) outs(%3028 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1231 = tensor.extract_slice %expanded_608[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1232 = tensor.extract_slice %extracted_slice_1231[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1233 = tensor.extract_slice %extracted_slice_1232[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1234 = tensor.extract_slice %expanded_610[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1235 = tensor.extract_slice %extracted_slice_1234[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1236 = tensor.extract_slice %extracted_slice_1235[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %3030 = tensor.empty() : tensor<1x80x128xf32>
    %3031 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1233 : tensor<1x1x80x128xf32>) outs(%3030 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3032 = tensor.empty() : tensor<80x128xf32>
    %3033 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3031 : tensor<1x80x128xf32>) outs(%3032 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3034 = tensor.empty() : tensor<1x80x128xf32>
    %3035 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1236 : tensor<1x1x80x128xf32>) outs(%3034 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3036 = tensor.empty() : tensor<80x128xf32>
    %3037 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3035 : tensor<1x80x128xf32>) outs(%3036 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3038 = tensor.empty() : tensor<1x80x128xf32>
    %3039 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3038 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3033[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3040 = "tosa.reshape"(%3039) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3041 = tensor.empty() : tensor<1x80x128xf32>
    %3042 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3041 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3037[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3043 = "tosa.reshape"(%3042) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3044 = "tosa.mul"(%3023, %3040) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1237 = tensor.extract_slice %3023[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1238 = tensor.extract_slice %3023[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3045 = tensor.empty() : tensor<1x32x80x64xf32>
    %3046 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1238 : tensor<1x32x80x64xf32>) outs(%3045 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3047 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1239 = tensor.insert_slice %3046 into %3047[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1240 = tensor.insert_slice %extracted_slice_1237 into %inserted_slice_1239[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3048 = "tosa.mul"(%inserted_slice_1240, %3043) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3049 = "tosa.add"(%3044, %3048) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3050 = "tosa.mul"(%3026, %3040) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1241 = tensor.extract_slice %3026[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1242 = tensor.extract_slice %3026[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3051 = tensor.empty() : tensor<1x32x80x64xf32>
    %3052 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1242 : tensor<1x32x80x64xf32>) outs(%3051 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3053 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1243 = tensor.insert_slice %3052 into %3053[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1244 = tensor.insert_slice %extracted_slice_1241 into %inserted_slice_1243[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3054 = "tosa.mul"(%inserted_slice_1244, %3043) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3055 = "tosa.add"(%3050, %3054) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3056 = tensor.empty() : tensor<1x32x128x80xf32>
    %3057 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3055 : tensor<1x32x80x128xf32>) outs(%3056 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %3058 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3059 = "tosa.add"(%3049, %3058) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3060 = "tosa.reshape"(%3059) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3061 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %3062 = "tosa.add"(%3057, %3061) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %3063 = "tosa.reshape"(%3062) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %3064 = "tosa.matmul"(%3060, %3063) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %3065 = "tosa.reshape"(%3064) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3066 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3067 = "tosa.reciprocal"(%3066) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3068 = "tosa.mul"(%3065, %3067) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3069 = "tosa.add"(%3068, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3070 = tensor.empty() : tensor<1x32x80x1xf32>
    %3071 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3070 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3072 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3069 : tensor<1x32x80x80xf32>) outs(%3070 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3073 = tensor.empty() : tensor<1x32x80x80xf32>
    %3074 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3069, %3072 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3073 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %3075 = tensor.empty() : tensor<1x32x80x1xf32>
    %3076 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3075 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3077 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3074 : tensor<1x32x80x80xf32>) outs(%3076 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3078 = tensor.empty() : tensor<1x32x80x80xf32>
    %3079 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3074, %3077 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3078 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %3080 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3081 = "tosa.add"(%3079, %3080) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3082 = "tosa.reshape"(%3081) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %3083 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3084 = "tosa.add"(%3029, %3083) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3085 = "tosa.reshape"(%3084) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3086 = "tosa.matmul"(%3082, %3085) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %3087 = "tosa.reshape"(%3086) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3088 = tensor.empty() : tensor<1x80x32x128xf32>
    %3089 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3087 : tensor<1x32x80x128xf32>) outs(%3088 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %3090 = "tosa.identity"(%3089) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %3091 = "tosa.reshape"(%3090) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %3092 = tensor.empty() : tensor<4096x4096xf32>
    %3093 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_394 : tensor<4096x4096xf32>) outs(%3092 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3094 = "tosa.reshape"(%3091) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1245 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3095 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3094, %3093 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1245 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3096 = "tosa.reshape"(%3095) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3097 = "tosa.add"(%2996, %3096) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3098 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1246 = arith.constant 2 : i32
    %3099 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3097 : tensor<1x80x4096xf32>) outs(%3098 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1246 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1247 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3100 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3099 : tensor<1x80x4096xf32>) outs(%cst_1247 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3101 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3102 = "tosa.add"(%3100, %3101) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3103 = "tosa.rsqrt"(%3102) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3104 = "tosa.mul"(%3097, %3103) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3105 = "tosa.reshape"(%extracted_slice_46) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3106 = "tosa.mul"(%3105, %3104) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3107 = tensor.empty() : tensor<4096x11008xf32>
    %3108 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_396 : tensor<11008x4096xf32>) outs(%3107 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3109 = "tosa.reshape"(%3106) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1248 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3110 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3109, %3108 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1248 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3111 = "tosa.reshape"(%3110) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3112 = tensor.empty() : tensor<1x80x11008xf32>
    %3113 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3111 : tensor<1x80x11008xf32>) outs(%3112 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %3114 = tensor.empty() : tensor<4096x11008xf32>
    %3115 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_398 : tensor<11008x4096xf32>) outs(%3114 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3116 = "tosa.reshape"(%3106) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1249 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3117 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3116, %3115 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1249 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3118 = "tosa.reshape"(%3117) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3119 = "tosa.mul"(%3113, %3118) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %3120 = tensor.empty() : tensor<11008x4096xf32>
    %3121 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_400 : tensor<4096x11008xf32>) outs(%3120 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %3122 = "tosa.reshape"(%3119) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1250 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3123 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3122, %3121 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1250 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3124 = "tosa.reshape"(%3123) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3125 = "tosa.add"(%3097, %3124) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3126 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1251 = arith.constant 2 : i32
    %3127 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3125 : tensor<1x80x4096xf32>) outs(%3126 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1251 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1252 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3128 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3127 : tensor<1x80x4096xf32>) outs(%cst_1252 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3129 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3130 = "tosa.add"(%3128, %3129) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3131 = "tosa.rsqrt"(%3130) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3132 = "tosa.mul"(%3125, %3131) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3133 = "tosa.reshape"(%extracted_slice_47) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3134 = "tosa.mul"(%3133, %3132) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3135 = tensor.empty() : tensor<4096x4096xf32>
    %3136 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_402 : tensor<4096x4096xf32>) outs(%3135 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3137 = "tosa.reshape"(%3134) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1253 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3138 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3137, %3136 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1253 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3139 = "tosa.reshape"(%3138) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3140 = tensor.empty() : tensor<4096x4096xf32>
    %3141 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_404 : tensor<4096x4096xf32>) outs(%3140 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3142 = "tosa.reshape"(%3134) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1254 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3143 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3142, %3141 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1254 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3144 = "tosa.reshape"(%3143) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3145 = tensor.empty() : tensor<4096x4096xf32>
    %3146 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_406 : tensor<4096x4096xf32>) outs(%3145 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3147 = "tosa.reshape"(%3134) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1255 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3148 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3147, %3146 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1255 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3149 = "tosa.reshape"(%3148) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3150 = "tosa.reshape"(%3139) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3151 = tensor.empty() : tensor<1x32x80x128xf32>
    %3152 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3150 : tensor<1x80x32x128xf32>) outs(%3151 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3153 = "tosa.reshape"(%3144) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3154 = tensor.empty() : tensor<1x32x80x128xf32>
    %3155 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3153 : tensor<1x80x32x128xf32>) outs(%3154 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3156 = "tosa.reshape"(%3149) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3157 = tensor.empty() : tensor<1x32x80x128xf32>
    %3158 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3156 : tensor<1x80x32x128xf32>) outs(%3157 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1256 = tensor.extract_slice %expanded_612[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1257 = tensor.extract_slice %extracted_slice_1256[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1258 = tensor.extract_slice %extracted_slice_1257[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1259 = tensor.extract_slice %expanded_614[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1260 = tensor.extract_slice %extracted_slice_1259[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1261 = tensor.extract_slice %extracted_slice_1260[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %3159 = tensor.empty() : tensor<1x80x128xf32>
    %3160 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1258 : tensor<1x1x80x128xf32>) outs(%3159 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3161 = tensor.empty() : tensor<80x128xf32>
    %3162 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3160 : tensor<1x80x128xf32>) outs(%3161 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3163 = tensor.empty() : tensor<1x80x128xf32>
    %3164 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1261 : tensor<1x1x80x128xf32>) outs(%3163 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3165 = tensor.empty() : tensor<80x128xf32>
    %3166 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3164 : tensor<1x80x128xf32>) outs(%3165 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3167 = tensor.empty() : tensor<1x80x128xf32>
    %3168 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3167 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3162[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3169 = "tosa.reshape"(%3168) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3170 = tensor.empty() : tensor<1x80x128xf32>
    %3171 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3170 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3166[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3172 = "tosa.reshape"(%3171) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3173 = "tosa.mul"(%3152, %3169) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1262 = tensor.extract_slice %3152[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1263 = tensor.extract_slice %3152[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3174 = tensor.empty() : tensor<1x32x80x64xf32>
    %3175 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1263 : tensor<1x32x80x64xf32>) outs(%3174 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3176 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1264 = tensor.insert_slice %3175 into %3176[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1265 = tensor.insert_slice %extracted_slice_1262 into %inserted_slice_1264[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3177 = "tosa.mul"(%inserted_slice_1265, %3172) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3178 = "tosa.add"(%3173, %3177) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3179 = "tosa.mul"(%3155, %3169) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1266 = tensor.extract_slice %3155[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1267 = tensor.extract_slice %3155[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3180 = tensor.empty() : tensor<1x32x80x64xf32>
    %3181 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1267 : tensor<1x32x80x64xf32>) outs(%3180 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3182 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1268 = tensor.insert_slice %3181 into %3182[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1269 = tensor.insert_slice %extracted_slice_1266 into %inserted_slice_1268[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3183 = "tosa.mul"(%inserted_slice_1269, %3172) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3184 = "tosa.add"(%3179, %3183) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3185 = tensor.empty() : tensor<1x32x128x80xf32>
    %3186 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3184 : tensor<1x32x80x128xf32>) outs(%3185 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %3187 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3188 = "tosa.add"(%3178, %3187) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3189 = "tosa.reshape"(%3188) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3190 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %3191 = "tosa.add"(%3186, %3190) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %3192 = "tosa.reshape"(%3191) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %3193 = "tosa.matmul"(%3189, %3192) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %3194 = "tosa.reshape"(%3193) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3195 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3196 = "tosa.reciprocal"(%3195) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3197 = "tosa.mul"(%3194, %3196) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3198 = "tosa.add"(%3197, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3199 = tensor.empty() : tensor<1x32x80x1xf32>
    %3200 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3199 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3201 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3198 : tensor<1x32x80x80xf32>) outs(%3199 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3202 = tensor.empty() : tensor<1x32x80x80xf32>
    %3203 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3198, %3201 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3202 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %3204 = tensor.empty() : tensor<1x32x80x1xf32>
    %3205 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3204 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3206 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3203 : tensor<1x32x80x80xf32>) outs(%3205 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3207 = tensor.empty() : tensor<1x32x80x80xf32>
    %3208 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3203, %3206 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3207 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %3209 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3210 = "tosa.add"(%3208, %3209) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3211 = "tosa.reshape"(%3210) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %3212 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3213 = "tosa.add"(%3158, %3212) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3214 = "tosa.reshape"(%3213) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3215 = "tosa.matmul"(%3211, %3214) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %3216 = "tosa.reshape"(%3215) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3217 = tensor.empty() : tensor<1x80x32x128xf32>
    %3218 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3216 : tensor<1x32x80x128xf32>) outs(%3217 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %3219 = "tosa.identity"(%3218) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %3220 = "tosa.reshape"(%3219) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %3221 = tensor.empty() : tensor<4096x4096xf32>
    %3222 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_408 : tensor<4096x4096xf32>) outs(%3221 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3223 = "tosa.reshape"(%3220) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1270 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3224 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3223, %3222 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1270 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3225 = "tosa.reshape"(%3224) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3226 = "tosa.add"(%3125, %3225) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3227 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1271 = arith.constant 2 : i32
    %3228 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3226 : tensor<1x80x4096xf32>) outs(%3227 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1271 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1272 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3229 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3228 : tensor<1x80x4096xf32>) outs(%cst_1272 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3230 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3231 = "tosa.add"(%3229, %3230) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3232 = "tosa.rsqrt"(%3231) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3233 = "tosa.mul"(%3226, %3232) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3234 = "tosa.reshape"(%extracted_slice_48) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3235 = "tosa.mul"(%3234, %3233) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3236 = tensor.empty() : tensor<4096x11008xf32>
    %3237 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_410 : tensor<11008x4096xf32>) outs(%3236 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3238 = "tosa.reshape"(%3235) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1273 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3239 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3238, %3237 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1273 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3240 = "tosa.reshape"(%3239) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3241 = tensor.empty() : tensor<1x80x11008xf32>
    %3242 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3240 : tensor<1x80x11008xf32>) outs(%3241 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %3243 = tensor.empty() : tensor<4096x11008xf32>
    %3244 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_412 : tensor<11008x4096xf32>) outs(%3243 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3245 = "tosa.reshape"(%3235) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1274 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3246 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3245, %3244 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1274 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3247 = "tosa.reshape"(%3246) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3248 = "tosa.mul"(%3242, %3247) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %3249 = tensor.empty() : tensor<11008x4096xf32>
    %3250 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_414 : tensor<4096x11008xf32>) outs(%3249 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %3251 = "tosa.reshape"(%3248) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1275 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3252 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3251, %3250 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1275 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3253 = "tosa.reshape"(%3252) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3254 = "tosa.add"(%3226, %3253) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3255 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1276 = arith.constant 2 : i32
    %3256 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3254 : tensor<1x80x4096xf32>) outs(%3255 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1276 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1277 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3257 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3256 : tensor<1x80x4096xf32>) outs(%cst_1277 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3258 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3259 = "tosa.add"(%3257, %3258) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3260 = "tosa.rsqrt"(%3259) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3261 = "tosa.mul"(%3254, %3260) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3262 = "tosa.reshape"(%extracted_slice_49) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3263 = "tosa.mul"(%3262, %3261) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3264 = tensor.empty() : tensor<4096x4096xf32>
    %3265 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_416 : tensor<4096x4096xf32>) outs(%3264 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3266 = "tosa.reshape"(%3263) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1278 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3267 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3266, %3265 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1278 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3268 = "tosa.reshape"(%3267) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3269 = tensor.empty() : tensor<4096x4096xf32>
    %3270 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_418 : tensor<4096x4096xf32>) outs(%3269 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3271 = "tosa.reshape"(%3263) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1279 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3272 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3271, %3270 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1279 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3273 = "tosa.reshape"(%3272) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3274 = tensor.empty() : tensor<4096x4096xf32>
    %3275 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_420 : tensor<4096x4096xf32>) outs(%3274 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3276 = "tosa.reshape"(%3263) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1280 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3277 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3276, %3275 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1280 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3278 = "tosa.reshape"(%3277) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3279 = "tosa.reshape"(%3268) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3280 = tensor.empty() : tensor<1x32x80x128xf32>
    %3281 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3279 : tensor<1x80x32x128xf32>) outs(%3280 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3282 = "tosa.reshape"(%3273) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3283 = tensor.empty() : tensor<1x32x80x128xf32>
    %3284 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3282 : tensor<1x80x32x128xf32>) outs(%3283 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3285 = "tosa.reshape"(%3278) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3286 = tensor.empty() : tensor<1x32x80x128xf32>
    %3287 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3285 : tensor<1x80x32x128xf32>) outs(%3286 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1281 = tensor.extract_slice %expanded_616[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1282 = tensor.extract_slice %extracted_slice_1281[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1283 = tensor.extract_slice %extracted_slice_1282[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1284 = tensor.extract_slice %expanded_618[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1285 = tensor.extract_slice %extracted_slice_1284[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1286 = tensor.extract_slice %extracted_slice_1285[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %3288 = tensor.empty() : tensor<1x80x128xf32>
    %3289 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1283 : tensor<1x1x80x128xf32>) outs(%3288 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3290 = tensor.empty() : tensor<80x128xf32>
    %3291 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3289 : tensor<1x80x128xf32>) outs(%3290 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3292 = tensor.empty() : tensor<1x80x128xf32>
    %3293 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1286 : tensor<1x1x80x128xf32>) outs(%3292 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3294 = tensor.empty() : tensor<80x128xf32>
    %3295 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3293 : tensor<1x80x128xf32>) outs(%3294 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3296 = tensor.empty() : tensor<1x80x128xf32>
    %3297 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3296 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3291[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3298 = "tosa.reshape"(%3297) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3299 = tensor.empty() : tensor<1x80x128xf32>
    %3300 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3299 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3295[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3301 = "tosa.reshape"(%3300) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3302 = "tosa.mul"(%3281, %3298) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1287 = tensor.extract_slice %3281[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1288 = tensor.extract_slice %3281[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3303 = tensor.empty() : tensor<1x32x80x64xf32>
    %3304 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1288 : tensor<1x32x80x64xf32>) outs(%3303 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3305 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1289 = tensor.insert_slice %3304 into %3305[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1290 = tensor.insert_slice %extracted_slice_1287 into %inserted_slice_1289[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3306 = "tosa.mul"(%inserted_slice_1290, %3301) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3307 = "tosa.add"(%3302, %3306) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3308 = "tosa.mul"(%3284, %3298) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1291 = tensor.extract_slice %3284[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1292 = tensor.extract_slice %3284[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3309 = tensor.empty() : tensor<1x32x80x64xf32>
    %3310 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1292 : tensor<1x32x80x64xf32>) outs(%3309 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3311 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1293 = tensor.insert_slice %3310 into %3311[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1294 = tensor.insert_slice %extracted_slice_1291 into %inserted_slice_1293[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3312 = "tosa.mul"(%inserted_slice_1294, %3301) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3313 = "tosa.add"(%3308, %3312) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3314 = tensor.empty() : tensor<1x32x128x80xf32>
    %3315 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3313 : tensor<1x32x80x128xf32>) outs(%3314 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %3316 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3317 = "tosa.add"(%3307, %3316) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3318 = "tosa.reshape"(%3317) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3319 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %3320 = "tosa.add"(%3315, %3319) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %3321 = "tosa.reshape"(%3320) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %3322 = "tosa.matmul"(%3318, %3321) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %3323 = "tosa.reshape"(%3322) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3324 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3325 = "tosa.reciprocal"(%3324) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3326 = "tosa.mul"(%3323, %3325) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3327 = "tosa.add"(%3326, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3328 = tensor.empty() : tensor<1x32x80x1xf32>
    %3329 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3328 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3330 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3327 : tensor<1x32x80x80xf32>) outs(%3328 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3331 = tensor.empty() : tensor<1x32x80x80xf32>
    %3332 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3327, %3330 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3331 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %3333 = tensor.empty() : tensor<1x32x80x1xf32>
    %3334 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3333 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3335 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3332 : tensor<1x32x80x80xf32>) outs(%3334 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3336 = tensor.empty() : tensor<1x32x80x80xf32>
    %3337 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3332, %3335 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3336 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %3338 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3339 = "tosa.add"(%3337, %3338) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3340 = "tosa.reshape"(%3339) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %3341 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3342 = "tosa.add"(%3287, %3341) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3343 = "tosa.reshape"(%3342) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3344 = "tosa.matmul"(%3340, %3343) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %3345 = "tosa.reshape"(%3344) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3346 = tensor.empty() : tensor<1x80x32x128xf32>
    %3347 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3345 : tensor<1x32x80x128xf32>) outs(%3346 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %3348 = "tosa.identity"(%3347) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %3349 = "tosa.reshape"(%3348) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %3350 = tensor.empty() : tensor<4096x4096xf32>
    %3351 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_422 : tensor<4096x4096xf32>) outs(%3350 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3352 = "tosa.reshape"(%3349) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1295 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3353 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3352, %3351 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1295 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3354 = "tosa.reshape"(%3353) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3355 = "tosa.add"(%3254, %3354) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3356 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1296 = arith.constant 2 : i32
    %3357 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3355 : tensor<1x80x4096xf32>) outs(%3356 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1296 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1297 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3358 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3357 : tensor<1x80x4096xf32>) outs(%cst_1297 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3359 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3360 = "tosa.add"(%3358, %3359) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3361 = "tosa.rsqrt"(%3360) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3362 = "tosa.mul"(%3355, %3361) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3363 = "tosa.reshape"(%extracted_slice_50) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3364 = "tosa.mul"(%3363, %3362) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3365 = tensor.empty() : tensor<4096x11008xf32>
    %3366 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_424 : tensor<11008x4096xf32>) outs(%3365 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3367 = "tosa.reshape"(%3364) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1298 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3368 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3367, %3366 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1298 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3369 = "tosa.reshape"(%3368) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3370 = tensor.empty() : tensor<1x80x11008xf32>
    %3371 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3369 : tensor<1x80x11008xf32>) outs(%3370 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %3372 = tensor.empty() : tensor<4096x11008xf32>
    %3373 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_426 : tensor<11008x4096xf32>) outs(%3372 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3374 = "tosa.reshape"(%3364) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1299 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3375 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3374, %3373 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1299 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3376 = "tosa.reshape"(%3375) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3377 = "tosa.mul"(%3371, %3376) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %3378 = tensor.empty() : tensor<11008x4096xf32>
    %3379 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_428 : tensor<4096x11008xf32>) outs(%3378 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %3380 = "tosa.reshape"(%3377) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1300 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3381 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3380, %3379 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1300 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3382 = "tosa.reshape"(%3381) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3383 = "tosa.add"(%3355, %3382) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3384 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1301 = arith.constant 2 : i32
    %3385 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3383 : tensor<1x80x4096xf32>) outs(%3384 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1301 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1302 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3386 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3385 : tensor<1x80x4096xf32>) outs(%cst_1302 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3387 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3388 = "tosa.add"(%3386, %3387) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3389 = "tosa.rsqrt"(%3388) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3390 = "tosa.mul"(%3383, %3389) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3391 = "tosa.reshape"(%extracted_slice_51) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3392 = "tosa.mul"(%3391, %3390) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3393 = tensor.empty() : tensor<4096x4096xf32>
    %3394 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_430 : tensor<4096x4096xf32>) outs(%3393 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3395 = "tosa.reshape"(%3392) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1303 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3396 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3395, %3394 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1303 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3397 = "tosa.reshape"(%3396) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3398 = tensor.empty() : tensor<4096x4096xf32>
    %3399 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_432 : tensor<4096x4096xf32>) outs(%3398 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3400 = "tosa.reshape"(%3392) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1304 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3401 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3400, %3399 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1304 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3402 = "tosa.reshape"(%3401) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3403 = tensor.empty() : tensor<4096x4096xf32>
    %3404 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_434 : tensor<4096x4096xf32>) outs(%3403 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3405 = "tosa.reshape"(%3392) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1305 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3406 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3405, %3404 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1305 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3407 = "tosa.reshape"(%3406) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3408 = "tosa.reshape"(%3397) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3409 = tensor.empty() : tensor<1x32x80x128xf32>
    %3410 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3408 : tensor<1x80x32x128xf32>) outs(%3409 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3411 = "tosa.reshape"(%3402) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3412 = tensor.empty() : tensor<1x32x80x128xf32>
    %3413 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3411 : tensor<1x80x32x128xf32>) outs(%3412 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3414 = "tosa.reshape"(%3407) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3415 = tensor.empty() : tensor<1x32x80x128xf32>
    %3416 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3414 : tensor<1x80x32x128xf32>) outs(%3415 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1306 = tensor.extract_slice %expanded_620[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1307 = tensor.extract_slice %extracted_slice_1306[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1308 = tensor.extract_slice %extracted_slice_1307[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1309 = tensor.extract_slice %expanded_622[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1310 = tensor.extract_slice %extracted_slice_1309[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1311 = tensor.extract_slice %extracted_slice_1310[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %3417 = tensor.empty() : tensor<1x80x128xf32>
    %3418 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1308 : tensor<1x1x80x128xf32>) outs(%3417 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3419 = tensor.empty() : tensor<80x128xf32>
    %3420 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3418 : tensor<1x80x128xf32>) outs(%3419 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3421 = tensor.empty() : tensor<1x80x128xf32>
    %3422 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1311 : tensor<1x1x80x128xf32>) outs(%3421 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3423 = tensor.empty() : tensor<80x128xf32>
    %3424 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3422 : tensor<1x80x128xf32>) outs(%3423 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3425 = tensor.empty() : tensor<1x80x128xf32>
    %3426 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3425 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3420[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3427 = "tosa.reshape"(%3426) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3428 = tensor.empty() : tensor<1x80x128xf32>
    %3429 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3428 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3424[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3430 = "tosa.reshape"(%3429) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3431 = "tosa.mul"(%3410, %3427) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1312 = tensor.extract_slice %3410[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1313 = tensor.extract_slice %3410[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3432 = tensor.empty() : tensor<1x32x80x64xf32>
    %3433 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1313 : tensor<1x32x80x64xf32>) outs(%3432 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3434 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1314 = tensor.insert_slice %3433 into %3434[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1315 = tensor.insert_slice %extracted_slice_1312 into %inserted_slice_1314[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3435 = "tosa.mul"(%inserted_slice_1315, %3430) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3436 = "tosa.add"(%3431, %3435) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3437 = "tosa.mul"(%3413, %3427) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1316 = tensor.extract_slice %3413[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1317 = tensor.extract_slice %3413[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3438 = tensor.empty() : tensor<1x32x80x64xf32>
    %3439 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1317 : tensor<1x32x80x64xf32>) outs(%3438 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3440 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1318 = tensor.insert_slice %3439 into %3440[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1319 = tensor.insert_slice %extracted_slice_1316 into %inserted_slice_1318[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3441 = "tosa.mul"(%inserted_slice_1319, %3430) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3442 = "tosa.add"(%3437, %3441) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3443 = tensor.empty() : tensor<1x32x128x80xf32>
    %3444 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3442 : tensor<1x32x80x128xf32>) outs(%3443 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %3445 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3446 = "tosa.add"(%3436, %3445) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3447 = "tosa.reshape"(%3446) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3448 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %3449 = "tosa.add"(%3444, %3448) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %3450 = "tosa.reshape"(%3449) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %3451 = "tosa.matmul"(%3447, %3450) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %3452 = "tosa.reshape"(%3451) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3453 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3454 = "tosa.reciprocal"(%3453) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3455 = "tosa.mul"(%3452, %3454) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3456 = "tosa.add"(%3455, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3457 = tensor.empty() : tensor<1x32x80x1xf32>
    %3458 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3457 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3459 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3456 : tensor<1x32x80x80xf32>) outs(%3457 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3460 = tensor.empty() : tensor<1x32x80x80xf32>
    %3461 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3456, %3459 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3460 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %3462 = tensor.empty() : tensor<1x32x80x1xf32>
    %3463 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3462 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3464 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3461 : tensor<1x32x80x80xf32>) outs(%3463 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3465 = tensor.empty() : tensor<1x32x80x80xf32>
    %3466 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3461, %3464 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3465 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %3467 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3468 = "tosa.add"(%3466, %3467) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3469 = "tosa.reshape"(%3468) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %3470 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3471 = "tosa.add"(%3416, %3470) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3472 = "tosa.reshape"(%3471) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3473 = "tosa.matmul"(%3469, %3472) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %3474 = "tosa.reshape"(%3473) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3475 = tensor.empty() : tensor<1x80x32x128xf32>
    %3476 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3474 : tensor<1x32x80x128xf32>) outs(%3475 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %3477 = "tosa.identity"(%3476) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %3478 = "tosa.reshape"(%3477) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %3479 = tensor.empty() : tensor<4096x4096xf32>
    %3480 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_436 : tensor<4096x4096xf32>) outs(%3479 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3481 = "tosa.reshape"(%3478) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1320 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3482 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3481, %3480 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1320 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3483 = "tosa.reshape"(%3482) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3484 = "tosa.add"(%3383, %3483) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3485 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1321 = arith.constant 2 : i32
    %3486 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3484 : tensor<1x80x4096xf32>) outs(%3485 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1321 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1322 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3487 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3486 : tensor<1x80x4096xf32>) outs(%cst_1322 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3488 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3489 = "tosa.add"(%3487, %3488) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3490 = "tosa.rsqrt"(%3489) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3491 = "tosa.mul"(%3484, %3490) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3492 = "tosa.reshape"(%extracted_slice_52) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3493 = "tosa.mul"(%3492, %3491) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3494 = tensor.empty() : tensor<4096x11008xf32>
    %3495 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_438 : tensor<11008x4096xf32>) outs(%3494 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3496 = "tosa.reshape"(%3493) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1323 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3497 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3496, %3495 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1323 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3498 = "tosa.reshape"(%3497) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3499 = tensor.empty() : tensor<1x80x11008xf32>
    %3500 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3498 : tensor<1x80x11008xf32>) outs(%3499 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %3501 = tensor.empty() : tensor<4096x11008xf32>
    %3502 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_440 : tensor<11008x4096xf32>) outs(%3501 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3503 = "tosa.reshape"(%3493) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1324 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3504 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3503, %3502 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1324 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3505 = "tosa.reshape"(%3504) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3506 = "tosa.mul"(%3500, %3505) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %3507 = tensor.empty() : tensor<11008x4096xf32>
    %3508 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_442 : tensor<4096x11008xf32>) outs(%3507 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %3509 = "tosa.reshape"(%3506) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1325 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3510 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3509, %3508 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1325 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3511 = "tosa.reshape"(%3510) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3512 = "tosa.add"(%3484, %3511) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3513 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1326 = arith.constant 2 : i32
    %3514 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3512 : tensor<1x80x4096xf32>) outs(%3513 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1326 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1327 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3515 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3514 : tensor<1x80x4096xf32>) outs(%cst_1327 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3516 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3517 = "tosa.add"(%3515, %3516) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3518 = "tosa.rsqrt"(%3517) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3519 = "tosa.mul"(%3512, %3518) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3520 = "tosa.reshape"(%extracted_slice_53) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3521 = "tosa.mul"(%3520, %3519) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3522 = tensor.empty() : tensor<4096x4096xf32>
    %3523 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_444 : tensor<4096x4096xf32>) outs(%3522 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3524 = "tosa.reshape"(%3521) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1328 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3525 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3524, %3523 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1328 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3526 = "tosa.reshape"(%3525) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3527 = tensor.empty() : tensor<4096x4096xf32>
    %3528 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_446 : tensor<4096x4096xf32>) outs(%3527 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3529 = "tosa.reshape"(%3521) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1329 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3530 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3529, %3528 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1329 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3531 = "tosa.reshape"(%3530) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3532 = tensor.empty() : tensor<4096x4096xf32>
    %3533 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_448 : tensor<4096x4096xf32>) outs(%3532 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3534 = "tosa.reshape"(%3521) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1330 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3535 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3534, %3533 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1330 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3536 = "tosa.reshape"(%3535) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3537 = "tosa.reshape"(%3526) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3538 = tensor.empty() : tensor<1x32x80x128xf32>
    %3539 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3537 : tensor<1x80x32x128xf32>) outs(%3538 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3540 = "tosa.reshape"(%3531) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3541 = tensor.empty() : tensor<1x32x80x128xf32>
    %3542 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3540 : tensor<1x80x32x128xf32>) outs(%3541 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3543 = "tosa.reshape"(%3536) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3544 = tensor.empty() : tensor<1x32x80x128xf32>
    %3545 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3543 : tensor<1x80x32x128xf32>) outs(%3544 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1331 = tensor.extract_slice %expanded_624[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1332 = tensor.extract_slice %extracted_slice_1331[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1333 = tensor.extract_slice %extracted_slice_1332[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1334 = tensor.extract_slice %expanded_626[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1335 = tensor.extract_slice %extracted_slice_1334[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1336 = tensor.extract_slice %extracted_slice_1335[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %3546 = tensor.empty() : tensor<1x80x128xf32>
    %3547 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1333 : tensor<1x1x80x128xf32>) outs(%3546 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3548 = tensor.empty() : tensor<80x128xf32>
    %3549 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3547 : tensor<1x80x128xf32>) outs(%3548 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3550 = tensor.empty() : tensor<1x80x128xf32>
    %3551 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1336 : tensor<1x1x80x128xf32>) outs(%3550 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3552 = tensor.empty() : tensor<80x128xf32>
    %3553 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3551 : tensor<1x80x128xf32>) outs(%3552 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3554 = tensor.empty() : tensor<1x80x128xf32>
    %3555 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3554 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3549[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3556 = "tosa.reshape"(%3555) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3557 = tensor.empty() : tensor<1x80x128xf32>
    %3558 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3557 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3553[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3559 = "tosa.reshape"(%3558) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3560 = "tosa.mul"(%3539, %3556) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1337 = tensor.extract_slice %3539[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1338 = tensor.extract_slice %3539[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3561 = tensor.empty() : tensor<1x32x80x64xf32>
    %3562 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1338 : tensor<1x32x80x64xf32>) outs(%3561 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3563 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1339 = tensor.insert_slice %3562 into %3563[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1340 = tensor.insert_slice %extracted_slice_1337 into %inserted_slice_1339[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3564 = "tosa.mul"(%inserted_slice_1340, %3559) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3565 = "tosa.add"(%3560, %3564) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3566 = "tosa.mul"(%3542, %3556) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1341 = tensor.extract_slice %3542[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1342 = tensor.extract_slice %3542[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3567 = tensor.empty() : tensor<1x32x80x64xf32>
    %3568 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1342 : tensor<1x32x80x64xf32>) outs(%3567 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3569 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1343 = tensor.insert_slice %3568 into %3569[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1344 = tensor.insert_slice %extracted_slice_1341 into %inserted_slice_1343[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3570 = "tosa.mul"(%inserted_slice_1344, %3559) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3571 = "tosa.add"(%3566, %3570) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3572 = tensor.empty() : tensor<1x32x128x80xf32>
    %3573 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3571 : tensor<1x32x80x128xf32>) outs(%3572 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %3574 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3575 = "tosa.add"(%3565, %3574) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3576 = "tosa.reshape"(%3575) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3577 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %3578 = "tosa.add"(%3573, %3577) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %3579 = "tosa.reshape"(%3578) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %3580 = "tosa.matmul"(%3576, %3579) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %3581 = "tosa.reshape"(%3580) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3582 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3583 = "tosa.reciprocal"(%3582) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3584 = "tosa.mul"(%3581, %3583) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3585 = "tosa.add"(%3584, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3586 = tensor.empty() : tensor<1x32x80x1xf32>
    %3587 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3586 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3588 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3585 : tensor<1x32x80x80xf32>) outs(%3586 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3589 = tensor.empty() : tensor<1x32x80x80xf32>
    %3590 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3585, %3588 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3589 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %3591 = tensor.empty() : tensor<1x32x80x1xf32>
    %3592 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3591 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3593 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3590 : tensor<1x32x80x80xf32>) outs(%3592 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3594 = tensor.empty() : tensor<1x32x80x80xf32>
    %3595 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3590, %3593 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3594 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %3596 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3597 = "tosa.add"(%3595, %3596) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3598 = "tosa.reshape"(%3597) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %3599 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3600 = "tosa.add"(%3545, %3599) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3601 = "tosa.reshape"(%3600) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3602 = "tosa.matmul"(%3598, %3601) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %3603 = "tosa.reshape"(%3602) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3604 = tensor.empty() : tensor<1x80x32x128xf32>
    %3605 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3603 : tensor<1x32x80x128xf32>) outs(%3604 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %3606 = "tosa.identity"(%3605) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %3607 = "tosa.reshape"(%3606) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %3608 = tensor.empty() : tensor<4096x4096xf32>
    %3609 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_450 : tensor<4096x4096xf32>) outs(%3608 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3610 = "tosa.reshape"(%3607) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1345 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3611 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3610, %3609 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1345 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3612 = "tosa.reshape"(%3611) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3613 = "tosa.add"(%3512, %3612) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3614 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1346 = arith.constant 2 : i32
    %3615 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3613 : tensor<1x80x4096xf32>) outs(%3614 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1346 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1347 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3616 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3615 : tensor<1x80x4096xf32>) outs(%cst_1347 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3617 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3618 = "tosa.add"(%3616, %3617) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3619 = "tosa.rsqrt"(%3618) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3620 = "tosa.mul"(%3613, %3619) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3621 = "tosa.reshape"(%extracted_slice_54) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3622 = "tosa.mul"(%3621, %3620) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3623 = tensor.empty() : tensor<4096x11008xf32>
    %3624 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_452 : tensor<11008x4096xf32>) outs(%3623 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3625 = "tosa.reshape"(%3622) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1348 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3626 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3625, %3624 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1348 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3627 = "tosa.reshape"(%3626) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3628 = tensor.empty() : tensor<1x80x11008xf32>
    %3629 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3627 : tensor<1x80x11008xf32>) outs(%3628 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %3630 = tensor.empty() : tensor<4096x11008xf32>
    %3631 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_454 : tensor<11008x4096xf32>) outs(%3630 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3632 = "tosa.reshape"(%3622) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1349 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3633 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3632, %3631 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1349 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3634 = "tosa.reshape"(%3633) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3635 = "tosa.mul"(%3629, %3634) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %3636 = tensor.empty() : tensor<11008x4096xf32>
    %3637 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_456 : tensor<4096x11008xf32>) outs(%3636 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %3638 = "tosa.reshape"(%3635) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1350 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3639 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3638, %3637 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1350 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3640 = "tosa.reshape"(%3639) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3641 = "tosa.add"(%3613, %3640) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3642 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1351 = arith.constant 2 : i32
    %3643 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3641 : tensor<1x80x4096xf32>) outs(%3642 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1351 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1352 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3644 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3643 : tensor<1x80x4096xf32>) outs(%cst_1352 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3645 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3646 = "tosa.add"(%3644, %3645) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3647 = "tosa.rsqrt"(%3646) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3648 = "tosa.mul"(%3641, %3647) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3649 = "tosa.reshape"(%extracted_slice_55) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3650 = "tosa.mul"(%3649, %3648) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3651 = tensor.empty() : tensor<4096x4096xf32>
    %3652 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_458 : tensor<4096x4096xf32>) outs(%3651 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3653 = "tosa.reshape"(%3650) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1353 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3654 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3653, %3652 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1353 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3655 = "tosa.reshape"(%3654) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3656 = tensor.empty() : tensor<4096x4096xf32>
    %3657 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_460 : tensor<4096x4096xf32>) outs(%3656 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3658 = "tosa.reshape"(%3650) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1354 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3659 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3658, %3657 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1354 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3660 = "tosa.reshape"(%3659) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3661 = tensor.empty() : tensor<4096x4096xf32>
    %3662 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_462 : tensor<4096x4096xf32>) outs(%3661 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3663 = "tosa.reshape"(%3650) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1355 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3664 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3663, %3662 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1355 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3665 = "tosa.reshape"(%3664) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3666 = "tosa.reshape"(%3655) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3667 = tensor.empty() : tensor<1x32x80x128xf32>
    %3668 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3666 : tensor<1x80x32x128xf32>) outs(%3667 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3669 = "tosa.reshape"(%3660) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3670 = tensor.empty() : tensor<1x32x80x128xf32>
    %3671 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3669 : tensor<1x80x32x128xf32>) outs(%3670 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3672 = "tosa.reshape"(%3665) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3673 = tensor.empty() : tensor<1x32x80x128xf32>
    %3674 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3672 : tensor<1x80x32x128xf32>) outs(%3673 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1356 = tensor.extract_slice %expanded_628[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1357 = tensor.extract_slice %extracted_slice_1356[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1358 = tensor.extract_slice %extracted_slice_1357[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1359 = tensor.extract_slice %expanded_630[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1360 = tensor.extract_slice %extracted_slice_1359[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1361 = tensor.extract_slice %extracted_slice_1360[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %3675 = tensor.empty() : tensor<1x80x128xf32>
    %3676 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1358 : tensor<1x1x80x128xf32>) outs(%3675 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3677 = tensor.empty() : tensor<80x128xf32>
    %3678 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3676 : tensor<1x80x128xf32>) outs(%3677 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3679 = tensor.empty() : tensor<1x80x128xf32>
    %3680 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1361 : tensor<1x1x80x128xf32>) outs(%3679 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3681 = tensor.empty() : tensor<80x128xf32>
    %3682 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3680 : tensor<1x80x128xf32>) outs(%3681 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3683 = tensor.empty() : tensor<1x80x128xf32>
    %3684 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3683 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3678[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3685 = "tosa.reshape"(%3684) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3686 = tensor.empty() : tensor<1x80x128xf32>
    %3687 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3686 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3682[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3688 = "tosa.reshape"(%3687) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3689 = "tosa.mul"(%3668, %3685) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1362 = tensor.extract_slice %3668[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1363 = tensor.extract_slice %3668[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3690 = tensor.empty() : tensor<1x32x80x64xf32>
    %3691 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1363 : tensor<1x32x80x64xf32>) outs(%3690 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3692 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1364 = tensor.insert_slice %3691 into %3692[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1365 = tensor.insert_slice %extracted_slice_1362 into %inserted_slice_1364[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3693 = "tosa.mul"(%inserted_slice_1365, %3688) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3694 = "tosa.add"(%3689, %3693) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3695 = "tosa.mul"(%3671, %3685) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1366 = tensor.extract_slice %3671[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1367 = tensor.extract_slice %3671[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3696 = tensor.empty() : tensor<1x32x80x64xf32>
    %3697 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1367 : tensor<1x32x80x64xf32>) outs(%3696 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3698 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1368 = tensor.insert_slice %3697 into %3698[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1369 = tensor.insert_slice %extracted_slice_1366 into %inserted_slice_1368[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3699 = "tosa.mul"(%inserted_slice_1369, %3688) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3700 = "tosa.add"(%3695, %3699) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3701 = tensor.empty() : tensor<1x32x128x80xf32>
    %3702 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3700 : tensor<1x32x80x128xf32>) outs(%3701 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %3703 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3704 = "tosa.add"(%3694, %3703) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3705 = "tosa.reshape"(%3704) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3706 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %3707 = "tosa.add"(%3702, %3706) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %3708 = "tosa.reshape"(%3707) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %3709 = "tosa.matmul"(%3705, %3708) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %3710 = "tosa.reshape"(%3709) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3711 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3712 = "tosa.reciprocal"(%3711) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3713 = "tosa.mul"(%3710, %3712) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3714 = "tosa.add"(%3713, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3715 = tensor.empty() : tensor<1x32x80x1xf32>
    %3716 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3715 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3717 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3714 : tensor<1x32x80x80xf32>) outs(%3715 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3718 = tensor.empty() : tensor<1x32x80x80xf32>
    %3719 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3714, %3717 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3718 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %3720 = tensor.empty() : tensor<1x32x80x1xf32>
    %3721 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3720 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3722 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3719 : tensor<1x32x80x80xf32>) outs(%3721 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3723 = tensor.empty() : tensor<1x32x80x80xf32>
    %3724 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3719, %3722 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3723 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %3725 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3726 = "tosa.add"(%3724, %3725) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3727 = "tosa.reshape"(%3726) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %3728 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3729 = "tosa.add"(%3674, %3728) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3730 = "tosa.reshape"(%3729) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3731 = "tosa.matmul"(%3727, %3730) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %3732 = "tosa.reshape"(%3731) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3733 = tensor.empty() : tensor<1x80x32x128xf32>
    %3734 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3732 : tensor<1x32x80x128xf32>) outs(%3733 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %3735 = "tosa.identity"(%3734) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %3736 = "tosa.reshape"(%3735) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %3737 = tensor.empty() : tensor<4096x4096xf32>
    %3738 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_464 : tensor<4096x4096xf32>) outs(%3737 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3739 = "tosa.reshape"(%3736) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1370 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3740 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3739, %3738 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1370 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3741 = "tosa.reshape"(%3740) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3742 = "tosa.add"(%3641, %3741) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3743 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1371 = arith.constant 2 : i32
    %3744 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3742 : tensor<1x80x4096xf32>) outs(%3743 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1371 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1372 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3745 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3744 : tensor<1x80x4096xf32>) outs(%cst_1372 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3746 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3747 = "tosa.add"(%3745, %3746) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3748 = "tosa.rsqrt"(%3747) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3749 = "tosa.mul"(%3742, %3748) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3750 = "tosa.reshape"(%extracted_slice_56) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3751 = "tosa.mul"(%3750, %3749) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3752 = tensor.empty() : tensor<4096x11008xf32>
    %3753 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_466 : tensor<11008x4096xf32>) outs(%3752 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3754 = "tosa.reshape"(%3751) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1373 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3755 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3754, %3753 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1373 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3756 = "tosa.reshape"(%3755) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3757 = tensor.empty() : tensor<1x80x11008xf32>
    %3758 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3756 : tensor<1x80x11008xf32>) outs(%3757 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %3759 = tensor.empty() : tensor<4096x11008xf32>
    %3760 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_468 : tensor<11008x4096xf32>) outs(%3759 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3761 = "tosa.reshape"(%3751) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1374 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3762 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3761, %3760 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1374 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3763 = "tosa.reshape"(%3762) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3764 = "tosa.mul"(%3758, %3763) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %3765 = tensor.empty() : tensor<11008x4096xf32>
    %3766 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_470 : tensor<4096x11008xf32>) outs(%3765 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %3767 = "tosa.reshape"(%3764) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1375 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3768 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3767, %3766 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1375 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3769 = "tosa.reshape"(%3768) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3770 = "tosa.add"(%3742, %3769) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3771 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1376 = arith.constant 2 : i32
    %3772 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3770 : tensor<1x80x4096xf32>) outs(%3771 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1376 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1377 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3773 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3772 : tensor<1x80x4096xf32>) outs(%cst_1377 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3774 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3775 = "tosa.add"(%3773, %3774) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3776 = "tosa.rsqrt"(%3775) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3777 = "tosa.mul"(%3770, %3776) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3778 = "tosa.reshape"(%extracted_slice_57) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3779 = "tosa.mul"(%3778, %3777) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3780 = tensor.empty() : tensor<4096x4096xf32>
    %3781 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_472 : tensor<4096x4096xf32>) outs(%3780 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3782 = "tosa.reshape"(%3779) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1378 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3783 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3782, %3781 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1378 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3784 = "tosa.reshape"(%3783) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3785 = tensor.empty() : tensor<4096x4096xf32>
    %3786 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_474 : tensor<4096x4096xf32>) outs(%3785 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3787 = "tosa.reshape"(%3779) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1379 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3788 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3787, %3786 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1379 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3789 = "tosa.reshape"(%3788) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3790 = tensor.empty() : tensor<4096x4096xf32>
    %3791 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_476 : tensor<4096x4096xf32>) outs(%3790 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3792 = "tosa.reshape"(%3779) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1380 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3793 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3792, %3791 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1380 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3794 = "tosa.reshape"(%3793) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3795 = "tosa.reshape"(%3784) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3796 = tensor.empty() : tensor<1x32x80x128xf32>
    %3797 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3795 : tensor<1x80x32x128xf32>) outs(%3796 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3798 = "tosa.reshape"(%3789) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3799 = tensor.empty() : tensor<1x32x80x128xf32>
    %3800 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3798 : tensor<1x80x32x128xf32>) outs(%3799 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3801 = "tosa.reshape"(%3794) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3802 = tensor.empty() : tensor<1x32x80x128xf32>
    %3803 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3801 : tensor<1x80x32x128xf32>) outs(%3802 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1381 = tensor.extract_slice %expanded_632[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1382 = tensor.extract_slice %extracted_slice_1381[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1383 = tensor.extract_slice %extracted_slice_1382[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1384 = tensor.extract_slice %expanded_634[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1385 = tensor.extract_slice %extracted_slice_1384[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1386 = tensor.extract_slice %extracted_slice_1385[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %3804 = tensor.empty() : tensor<1x80x128xf32>
    %3805 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1383 : tensor<1x1x80x128xf32>) outs(%3804 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3806 = tensor.empty() : tensor<80x128xf32>
    %3807 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3805 : tensor<1x80x128xf32>) outs(%3806 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3808 = tensor.empty() : tensor<1x80x128xf32>
    %3809 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1386 : tensor<1x1x80x128xf32>) outs(%3808 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3810 = tensor.empty() : tensor<80x128xf32>
    %3811 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3809 : tensor<1x80x128xf32>) outs(%3810 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3812 = tensor.empty() : tensor<1x80x128xf32>
    %3813 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3812 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3807[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3814 = "tosa.reshape"(%3813) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3815 = tensor.empty() : tensor<1x80x128xf32>
    %3816 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3815 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3811[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3817 = "tosa.reshape"(%3816) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3818 = "tosa.mul"(%3797, %3814) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1387 = tensor.extract_slice %3797[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1388 = tensor.extract_slice %3797[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3819 = tensor.empty() : tensor<1x32x80x64xf32>
    %3820 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1388 : tensor<1x32x80x64xf32>) outs(%3819 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3821 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1389 = tensor.insert_slice %3820 into %3821[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1390 = tensor.insert_slice %extracted_slice_1387 into %inserted_slice_1389[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3822 = "tosa.mul"(%inserted_slice_1390, %3817) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3823 = "tosa.add"(%3818, %3822) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3824 = "tosa.mul"(%3800, %3814) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1391 = tensor.extract_slice %3800[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1392 = tensor.extract_slice %3800[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3825 = tensor.empty() : tensor<1x32x80x64xf32>
    %3826 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1392 : tensor<1x32x80x64xf32>) outs(%3825 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3827 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1393 = tensor.insert_slice %3826 into %3827[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1394 = tensor.insert_slice %extracted_slice_1391 into %inserted_slice_1393[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3828 = "tosa.mul"(%inserted_slice_1394, %3817) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3829 = "tosa.add"(%3824, %3828) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3830 = tensor.empty() : tensor<1x32x128x80xf32>
    %3831 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3829 : tensor<1x32x80x128xf32>) outs(%3830 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %3832 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3833 = "tosa.add"(%3823, %3832) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3834 = "tosa.reshape"(%3833) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3835 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %3836 = "tosa.add"(%3831, %3835) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %3837 = "tosa.reshape"(%3836) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %3838 = "tosa.matmul"(%3834, %3837) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %3839 = "tosa.reshape"(%3838) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3840 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3841 = "tosa.reciprocal"(%3840) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3842 = "tosa.mul"(%3839, %3841) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3843 = "tosa.add"(%3842, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3844 = tensor.empty() : tensor<1x32x80x1xf32>
    %3845 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3844 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3846 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3843 : tensor<1x32x80x80xf32>) outs(%3844 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3847 = tensor.empty() : tensor<1x32x80x80xf32>
    %3848 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3843, %3846 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3847 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %3849 = tensor.empty() : tensor<1x32x80x1xf32>
    %3850 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3849 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3851 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3848 : tensor<1x32x80x80xf32>) outs(%3850 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3852 = tensor.empty() : tensor<1x32x80x80xf32>
    %3853 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3848, %3851 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3852 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %3854 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3855 = "tosa.add"(%3853, %3854) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3856 = "tosa.reshape"(%3855) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %3857 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3858 = "tosa.add"(%3803, %3857) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3859 = "tosa.reshape"(%3858) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3860 = "tosa.matmul"(%3856, %3859) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %3861 = "tosa.reshape"(%3860) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3862 = tensor.empty() : tensor<1x80x32x128xf32>
    %3863 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3861 : tensor<1x32x80x128xf32>) outs(%3862 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %3864 = "tosa.identity"(%3863) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %3865 = "tosa.reshape"(%3864) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %3866 = tensor.empty() : tensor<4096x4096xf32>
    %3867 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_478 : tensor<4096x4096xf32>) outs(%3866 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3868 = "tosa.reshape"(%3865) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1395 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3869 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3868, %3867 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1395 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3870 = "tosa.reshape"(%3869) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3871 = "tosa.add"(%3770, %3870) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3872 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1396 = arith.constant 2 : i32
    %3873 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3871 : tensor<1x80x4096xf32>) outs(%3872 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1396 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1397 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3874 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3873 : tensor<1x80x4096xf32>) outs(%cst_1397 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3875 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3876 = "tosa.add"(%3874, %3875) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3877 = "tosa.rsqrt"(%3876) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3878 = "tosa.mul"(%3871, %3877) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3879 = "tosa.reshape"(%extracted_slice_58) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3880 = "tosa.mul"(%3879, %3878) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3881 = tensor.empty() : tensor<4096x11008xf32>
    %3882 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_480 : tensor<11008x4096xf32>) outs(%3881 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3883 = "tosa.reshape"(%3880) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1398 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3884 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3883, %3882 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1398 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3885 = "tosa.reshape"(%3884) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3886 = tensor.empty() : tensor<1x80x11008xf32>
    %3887 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3885 : tensor<1x80x11008xf32>) outs(%3886 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %3888 = tensor.empty() : tensor<4096x11008xf32>
    %3889 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_482 : tensor<11008x4096xf32>) outs(%3888 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %3890 = "tosa.reshape"(%3880) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1399 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %3891 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3890, %3889 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1399 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %3892 = "tosa.reshape"(%3891) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %3893 = "tosa.mul"(%3887, %3892) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %3894 = tensor.empty() : tensor<11008x4096xf32>
    %3895 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_484 : tensor<4096x11008xf32>) outs(%3894 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %3896 = "tosa.reshape"(%3893) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1400 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3897 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3896, %3895 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1400 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3898 = "tosa.reshape"(%3897) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3899 = "tosa.add"(%3871, %3898) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3900 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1401 = arith.constant 2 : i32
    %3901 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3899 : tensor<1x80x4096xf32>) outs(%3900 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1401 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1402 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %3902 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%3901 : tensor<1x80x4096xf32>) outs(%cst_1402 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %3903 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %3904 = "tosa.add"(%3902, %3903) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3905 = "tosa.rsqrt"(%3904) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %3906 = "tosa.mul"(%3899, %3905) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %3907 = "tosa.reshape"(%extracted_slice_59) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %3908 = "tosa.mul"(%3907, %3906) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %3909 = tensor.empty() : tensor<4096x4096xf32>
    %3910 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_486 : tensor<4096x4096xf32>) outs(%3909 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3911 = "tosa.reshape"(%3908) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1403 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3912 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3911, %3910 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1403 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3913 = "tosa.reshape"(%3912) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3914 = tensor.empty() : tensor<4096x4096xf32>
    %3915 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_488 : tensor<4096x4096xf32>) outs(%3914 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3916 = "tosa.reshape"(%3908) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1404 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3917 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3916, %3915 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1404 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3918 = "tosa.reshape"(%3917) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3919 = tensor.empty() : tensor<4096x4096xf32>
    %3920 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_490 : tensor<4096x4096xf32>) outs(%3919 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3921 = "tosa.reshape"(%3908) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1405 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3922 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3921, %3920 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1405 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3923 = "tosa.reshape"(%3922) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %3924 = "tosa.reshape"(%3913) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3925 = tensor.empty() : tensor<1x32x80x128xf32>
    %3926 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3924 : tensor<1x80x32x128xf32>) outs(%3925 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3927 = "tosa.reshape"(%3918) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3928 = tensor.empty() : tensor<1x32x80x128xf32>
    %3929 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3927 : tensor<1x80x32x128xf32>) outs(%3928 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %3930 = "tosa.reshape"(%3923) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %3931 = tensor.empty() : tensor<1x32x80x128xf32>
    %3932 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3930 : tensor<1x80x32x128xf32>) outs(%3931 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1406 = tensor.extract_slice %expanded_636[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1407 = tensor.extract_slice %extracted_slice_1406[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1408 = tensor.extract_slice %extracted_slice_1407[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1409 = tensor.extract_slice %expanded_638[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1410 = tensor.extract_slice %extracted_slice_1409[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1411 = tensor.extract_slice %extracted_slice_1410[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %3933 = tensor.empty() : tensor<1x80x128xf32>
    %3934 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1408 : tensor<1x1x80x128xf32>) outs(%3933 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3935 = tensor.empty() : tensor<80x128xf32>
    %3936 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3934 : tensor<1x80x128xf32>) outs(%3935 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3937 = tensor.empty() : tensor<1x80x128xf32>
    %3938 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1411 : tensor<1x1x80x128xf32>) outs(%3937 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %3939 = tensor.empty() : tensor<80x128xf32>
    %3940 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%3938 : tensor<1x80x128xf32>) outs(%3939 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %3941 = tensor.empty() : tensor<1x80x128xf32>
    %3942 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3941 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3936[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3943 = "tosa.reshape"(%3942) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3944 = tensor.empty() : tensor<1x80x128xf32>
    %3945 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%3944 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %3940[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %3946 = "tosa.reshape"(%3945) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %3947 = "tosa.mul"(%3926, %3943) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1412 = tensor.extract_slice %3926[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1413 = tensor.extract_slice %3926[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3948 = tensor.empty() : tensor<1x32x80x64xf32>
    %3949 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1413 : tensor<1x32x80x64xf32>) outs(%3948 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3950 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1414 = tensor.insert_slice %3949 into %3950[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1415 = tensor.insert_slice %extracted_slice_1412 into %inserted_slice_1414[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3951 = "tosa.mul"(%inserted_slice_1415, %3946) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3952 = "tosa.add"(%3947, %3951) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3953 = "tosa.mul"(%3929, %3943) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1416 = tensor.extract_slice %3929[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1417 = tensor.extract_slice %3929[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %3954 = tensor.empty() : tensor<1x32x80x64xf32>
    %3955 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1417 : tensor<1x32x80x64xf32>) outs(%3954 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %3956 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1418 = tensor.insert_slice %3955 into %3956[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1419 = tensor.insert_slice %extracted_slice_1416 into %inserted_slice_1418[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %3957 = "tosa.mul"(%inserted_slice_1419, %3946) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3958 = "tosa.add"(%3953, %3957) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3959 = tensor.empty() : tensor<1x32x128x80xf32>
    %3960 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3958 : tensor<1x32x80x128xf32>) outs(%3959 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %3961 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3962 = "tosa.add"(%3952, %3961) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3963 = "tosa.reshape"(%3962) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3964 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %3965 = "tosa.add"(%3960, %3964) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %3966 = "tosa.reshape"(%3965) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %3967 = "tosa.matmul"(%3963, %3966) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %3968 = "tosa.reshape"(%3967) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3969 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3970 = "tosa.reciprocal"(%3969) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3971 = "tosa.mul"(%3968, %3970) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3972 = "tosa.add"(%3971, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3973 = tensor.empty() : tensor<1x32x80x1xf32>
    %3974 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3973 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3975 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3972 : tensor<1x32x80x80xf32>) outs(%3973 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3976 = tensor.empty() : tensor<1x32x80x80xf32>
    %3977 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3972, %3975 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3976 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %3978 = tensor.empty() : tensor<1x32x80x1xf32>
    %3979 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3978 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %3980 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3977 : tensor<1x32x80x80xf32>) outs(%3979 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %3981 = tensor.empty() : tensor<1x32x80x80xf32>
    %3982 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3977, %3980 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%3981 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %3983 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %3984 = "tosa.add"(%3982, %3983) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %3985 = "tosa.reshape"(%3984) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %3986 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %3987 = "tosa.add"(%3932, %3986) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3988 = "tosa.reshape"(%3987) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %3989 = "tosa.matmul"(%3985, %3988) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %3990 = "tosa.reshape"(%3989) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %3991 = tensor.empty() : tensor<1x80x32x128xf32>
    %3992 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3990 : tensor<1x32x80x128xf32>) outs(%3991 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %3993 = "tosa.identity"(%3992) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %3994 = "tosa.reshape"(%3993) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %3995 = tensor.empty() : tensor<4096x4096xf32>
    %3996 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_492 : tensor<4096x4096xf32>) outs(%3995 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %3997 = "tosa.reshape"(%3994) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1420 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %3998 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3997, %3996 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1420 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %3999 = "tosa.reshape"(%3998) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %4000 = "tosa.add"(%3899, %3999) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %4001 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1421 = arith.constant 2 : i32
    %4002 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4000 : tensor<1x80x4096xf32>) outs(%4001 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1421 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1422 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %4003 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4002 : tensor<1x80x4096xf32>) outs(%cst_1422 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %4004 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %4005 = "tosa.add"(%4003, %4004) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %4006 = "tosa.rsqrt"(%4005) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %4007 = "tosa.mul"(%4000, %4006) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %4008 = "tosa.reshape"(%extracted_slice_60) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4009 = "tosa.mul"(%4008, %4007) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %4010 = tensor.empty() : tensor<4096x11008xf32>
    %4011 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_494 : tensor<11008x4096xf32>) outs(%4010 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %4012 = "tosa.reshape"(%4009) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1423 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %4013 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4012, %4011 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1423 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %4014 = "tosa.reshape"(%4013) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %4015 = tensor.empty() : tensor<1x80x11008xf32>
    %4016 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4014 : tensor<1x80x11008xf32>) outs(%4015 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %4017 = tensor.empty() : tensor<4096x11008xf32>
    %4018 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_496 : tensor<11008x4096xf32>) outs(%4017 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %4019 = "tosa.reshape"(%4009) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1424 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %4020 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4019, %4018 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1424 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %4021 = "tosa.reshape"(%4020) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %4022 = "tosa.mul"(%4016, %4021) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %4023 = tensor.empty() : tensor<11008x4096xf32>
    %4024 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_498 : tensor<4096x11008xf32>) outs(%4023 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %4025 = "tosa.reshape"(%4022) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1425 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %4026 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4025, %4024 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1425 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %4027 = "tosa.reshape"(%4026) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %4028 = "tosa.add"(%4000, %4027) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %4029 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1426 = arith.constant 2 : i32
    %4030 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4028 : tensor<1x80x4096xf32>) outs(%4029 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1426 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1427 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %4031 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4030 : tensor<1x80x4096xf32>) outs(%cst_1427 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %4032 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %4033 = "tosa.add"(%4031, %4032) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %4034 = "tosa.rsqrt"(%4033) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %4035 = "tosa.mul"(%4028, %4034) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %4036 = "tosa.reshape"(%extracted_slice_61) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4037 = "tosa.mul"(%4036, %4035) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %4038 = tensor.empty() : tensor<4096x4096xf32>
    %4039 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_500 : tensor<4096x4096xf32>) outs(%4038 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %4040 = "tosa.reshape"(%4037) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1428 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %4041 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4040, %4039 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1428 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %4042 = "tosa.reshape"(%4041) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %4043 = tensor.empty() : tensor<4096x4096xf32>
    %4044 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_502 : tensor<4096x4096xf32>) outs(%4043 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %4045 = "tosa.reshape"(%4037) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1429 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %4046 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4045, %4044 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1429 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %4047 = "tosa.reshape"(%4046) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %4048 = tensor.empty() : tensor<4096x4096xf32>
    %4049 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_504 : tensor<4096x4096xf32>) outs(%4048 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %4050 = "tosa.reshape"(%4037) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1430 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %4051 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4050, %4049 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1430 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %4052 = "tosa.reshape"(%4051) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %4053 = "tosa.reshape"(%4042) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %4054 = tensor.empty() : tensor<1x32x80x128xf32>
    %4055 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4053 : tensor<1x80x32x128xf32>) outs(%4054 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %4056 = "tosa.reshape"(%4047) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %4057 = tensor.empty() : tensor<1x32x80x128xf32>
    %4058 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4056 : tensor<1x80x32x128xf32>) outs(%4057 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %4059 = "tosa.reshape"(%4052) {new_shape = array<i64: 1, 80, 32, 128>} : (tensor<1x80x4096xf32>) -> tensor<1x80x32x128xf32>
    %4060 = tensor.empty() : tensor<1x32x80x128xf32>
    %4061 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4059 : tensor<1x80x32x128xf32>) outs(%4060 : tensor<1x32x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x80x128xf32>
    %extracted_slice_1431 = tensor.extract_slice %expanded_640[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1432 = tensor.extract_slice %extracted_slice_1431[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1433 = tensor.extract_slice %extracted_slice_1432[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %extracted_slice_1434 = tensor.extract_slice %expanded_642[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1435 = tensor.extract_slice %extracted_slice_1434[0, 0, 0, 0] [1, 1, 2048, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x2048x128xf32>
    %extracted_slice_1436 = tensor.extract_slice %extracted_slice_1435[0, 0, 0, 0] [1, 1, 80, 128] [1, 1, 1, 1] : tensor<1x1x2048x128xf32> to tensor<1x1x80x128xf32>
    %4062 = tensor.empty() : tensor<1x80x128xf32>
    %4063 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1433 : tensor<1x1x80x128xf32>) outs(%4062 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %4064 = tensor.empty() : tensor<80x128xf32>
    %4065 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4063 : tensor<1x80x128xf32>) outs(%4064 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %4066 = tensor.empty() : tensor<1x80x128xf32>
    %4067 = linalg.generic {indexing_maps = [#map10, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_1436 : tensor<1x1x80x128xf32>) outs(%4066 : tensor<1x80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x128xf32>
    %4068 = tensor.empty() : tensor<80x128xf32>
    %4069 = linalg.generic {indexing_maps = [#map11, #map3], iterator_types = ["parallel", "parallel"]} ins(%4067 : tensor<1x80x128xf32>) outs(%4068 : tensor<80x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<80x128xf32>
    %4070 = tensor.empty() : tensor<1x80x128xf32>
    %4071 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%4070 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %4065[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %4072 = "tosa.reshape"(%4071) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %4073 = tensor.empty() : tensor<1x80x128xf32>
    %4074 = linalg.generic {indexing_maps = [#map2, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<1x80xi64>) outs(%4073 : tensor<1x80x128xf32>) {
    ^bb0(%in: i64, %out: f32):
      %4172 = arith.index_cast %in : i64 to index
      %4173 = linalg.index 2 : index
      %extracted = tensor.extract %4069[%4172, %4173] : tensor<80x128xf32>
      linalg.yield %extracted : f32
    } -> tensor<1x80x128xf32>
    %4075 = "tosa.reshape"(%4074) {new_shape = array<i64: 1, 1, 80, 128>} : (tensor<1x80x128xf32>) -> tensor<1x1x80x128xf32>
    %4076 = "tosa.mul"(%4055, %4072) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1437 = tensor.extract_slice %4055[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1438 = tensor.extract_slice %4055[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %4077 = tensor.empty() : tensor<1x32x80x64xf32>
    %4078 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1438 : tensor<1x32x80x64xf32>) outs(%4077 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %4079 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1439 = tensor.insert_slice %4078 into %4079[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1440 = tensor.insert_slice %extracted_slice_1437 into %inserted_slice_1439[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %4080 = "tosa.mul"(%inserted_slice_1440, %4075) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %4081 = "tosa.add"(%4076, %4080) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %4082 = "tosa.mul"(%4058, %4072) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %extracted_slice_1441 = tensor.extract_slice %4058[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %extracted_slice_1442 = tensor.extract_slice %4058[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x128xf32> to tensor<1x32x80x64xf32>
    %4083 = tensor.empty() : tensor<1x32x80x64xf32>
    %4084 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_1442 : tensor<1x32x80x64xf32>) outs(%4083 : tensor<1x32x80x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x64xf32>
    %4085 = tensor.empty() : tensor<1x32x80x128xf32>
    %inserted_slice_1443 = tensor.insert_slice %4084 into %4085[0, 0, 0, 0] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %inserted_slice_1444 = tensor.insert_slice %extracted_slice_1441 into %inserted_slice_1443[0, 0, 0, 64] [1, 32, 80, 64] [1, 1, 1, 1] : tensor<1x32x80x64xf32> into tensor<1x32x80x128xf32>
    %4086 = "tosa.mul"(%inserted_slice_1444, %4075) {shift = 0 : i32} : (tensor<1x32x80x128xf32>, tensor<1x1x80x128xf32>) -> tensor<1x32x80x128xf32>
    %4087 = "tosa.add"(%4082, %4086) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %4088 = tensor.empty() : tensor<1x32x128x80xf32>
    %4089 = linalg.generic {indexing_maps = [#map12, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4087 : tensor<1x32x80x128xf32>) outs(%4088 : tensor<1x32x128x80xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x128x80xf32>
    %4090 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %4091 = "tosa.add"(%4081, %4090) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %4092 = "tosa.reshape"(%4091) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %4093 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x128x80xf32>} : () -> tensor<1x32x128x80xf32>
    %4094 = "tosa.add"(%4089, %4093) : (tensor<1x32x128x80xf32>, tensor<1x32x128x80xf32>) -> tensor<1x32x128x80xf32>
    %4095 = "tosa.reshape"(%4094) {new_shape = array<i64: 32, 128, 80>} : (tensor<1x32x128x80xf32>) -> tensor<32x128x80xf32>
    %4096 = "tosa.matmul"(%4092, %4095) : (tensor<32x80x128xf32>, tensor<32x128x80xf32>) -> tensor<32x80x80xf32>
    %4097 = "tosa.reshape"(%4096) {new_shape = array<i64: 1, 32, 80, 80>} : (tensor<32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %4098 = "tosa.const"() {value = dense<11.3137083> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %4099 = "tosa.reciprocal"(%4098) : (tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %4100 = "tosa.mul"(%4097, %4099) {shift = 0 : i32} : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %4101 = "tosa.add"(%4100, %29) : (tensor<1x32x80x80xf32>, tensor<1x1x80x80xf32>) -> tensor<1x32x80x80xf32>
    %4102 = tensor.empty() : tensor<1x32x80x1xf32>
    %4103 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4102 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0xFF800000 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %4104 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4101 : tensor<1x32x80x80xf32>) outs(%4102 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.maxf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %4105 = tensor.empty() : tensor<1x32x80x80xf32>
    %4106 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4101, %4104 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%4105 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.subf %in, %in_1454 : f32
      %4173 = math.exp %4172 : f32
      linalg.yield %4173 : f32
    } -> tensor<1x32x80x80xf32>
    %4107 = tensor.empty() : tensor<1x32x80x1xf32>
    %4108 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4107 : tensor<1x32x80x1xf32>) {
    ^bb0(%out: f32):
      %cst_1454 = arith.constant 0.000000e+00 : f32
      linalg.yield %cst_1454 : f32
    } -> tensor<1x32x80x1xf32>
    %4109 = linalg.generic {indexing_maps = [#map4, #map13], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%4106 : tensor<1x32x80x80xf32>) outs(%4108 : tensor<1x32x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.addf %in, %out : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x1xf32>
    %4110 = tensor.empty() : tensor<1x32x80x80xf32>
    %4111 = linalg.generic {indexing_maps = [#map4, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4106, %4109 : tensor<1x32x80x80xf32>, tensor<1x32x80x1xf32>) outs(%4110 : tensor<1x32x80x80xf32>) {
    ^bb0(%in: f32, %in_1454: f32, %out: f32):
      %4172 = arith.divf %in, %in_1454 : f32
      linalg.yield %4172 : f32
    } -> tensor<1x32x80x80xf32>
    %4112 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x80xf32>} : () -> tensor<1x32x80x80xf32>
    %4113 = "tosa.add"(%4111, %4112) : (tensor<1x32x80x80xf32>, tensor<1x32x80x80xf32>) -> tensor<1x32x80x80xf32>
    %4114 = "tosa.reshape"(%4113) {new_shape = array<i64: 32, 80, 80>} : (tensor<1x32x80x80xf32>) -> tensor<32x80x80xf32>
    %4115 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x32x80x128xf32>} : () -> tensor<1x32x80x128xf32>
    %4116 = "tosa.add"(%4061, %4115) : (tensor<1x32x80x128xf32>, tensor<1x32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %4117 = "tosa.reshape"(%4116) {new_shape = array<i64: 32, 80, 128>} : (tensor<1x32x80x128xf32>) -> tensor<32x80x128xf32>
    %4118 = "tosa.matmul"(%4114, %4117) : (tensor<32x80x80xf32>, tensor<32x80x128xf32>) -> tensor<32x80x128xf32>
    %4119 = "tosa.reshape"(%4118) {new_shape = array<i64: 1, 32, 80, 128>} : (tensor<32x80x128xf32>) -> tensor<1x32x80x128xf32>
    %4120 = tensor.empty() : tensor<1x80x32x128xf32>
    %4121 = linalg.generic {indexing_maps = [#map9, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4119 : tensor<1x32x80x128xf32>) outs(%4120 : tensor<1x80x32x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x80x32x128xf32>
    %4122 = "tosa.identity"(%4121) : (tensor<1x80x32x128xf32>) -> tensor<1x80x32x128xf32>
    %4123 = "tosa.reshape"(%4122) {new_shape = array<i64: 1, 80, 4096>} : (tensor<1x80x32x128xf32>) -> tensor<1x80x4096xf32>
    %4124 = tensor.empty() : tensor<4096x4096xf32>
    %4125 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_506 : tensor<4096x4096xf32>) outs(%4124 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x4096xf32>
    %4126 = "tosa.reshape"(%4123) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1445 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %4127 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4126, %4125 : tensor<80x4096xf32>, tensor<4096x4096xf32>) outs(%cst_1445 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %4128 = "tosa.reshape"(%4127) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %4129 = "tosa.add"(%4028, %4128) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %4130 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1446 = arith.constant 2 : i32
    %4131 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4129 : tensor<1x80x4096xf32>) outs(%4130 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1446 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1447 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %4132 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4131 : tensor<1x80x4096xf32>) outs(%cst_1447 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %4133 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %4134 = "tosa.add"(%4132, %4133) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %4135 = "tosa.rsqrt"(%4134) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %4136 = "tosa.mul"(%4129, %4135) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %4137 = "tosa.reshape"(%extracted_slice_62) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4138 = "tosa.mul"(%4137, %4136) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %4139 = tensor.empty() : tensor<4096x11008xf32>
    %4140 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_508 : tensor<11008x4096xf32>) outs(%4139 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %4141 = "tosa.reshape"(%4138) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1448 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %4142 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4141, %4140 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1448 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %4143 = "tosa.reshape"(%4142) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %4144 = tensor.empty() : tensor<1x80x11008xf32>
    %4145 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4143 : tensor<1x80x11008xf32>) outs(%4144 : tensor<1x80x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = arith.negf %in : f32
      %4173 = math.exp %4172 : f32
      %cst_1454 = arith.constant 1.000000e+00 : f32
      %4174 = arith.addf %cst_1454, %4173 : f32
      %4175 = arith.divf %in, %4174 : f32
      linalg.yield %4175 : f32
    } -> tensor<1x80x11008xf32>
    %4146 = tensor.empty() : tensor<4096x11008xf32>
    %4147 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_510 : tensor<11008x4096xf32>) outs(%4146 : tensor<4096x11008xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x11008xf32>
    %4148 = "tosa.reshape"(%4138) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1449 = arith.constant dense<0.000000e+00> : tensor<80x11008xf32>
    %4149 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4148, %4147 : tensor<80x4096xf32>, tensor<4096x11008xf32>) outs(%cst_1449 : tensor<80x11008xf32>) -> tensor<80x11008xf32>
    %4150 = "tosa.reshape"(%4149) {new_shape = array<i64: 1, 80, 11008>} : (tensor<80x11008xf32>) -> tensor<1x80x11008xf32>
    %4151 = "tosa.mul"(%4145, %4150) {shift = 0 : i32} : (tensor<1x80x11008xf32>, tensor<1x80x11008xf32>) -> tensor<1x80x11008xf32>
    %4152 = tensor.empty() : tensor<11008x4096xf32>
    %4153 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_512 : tensor<4096x11008xf32>) outs(%4152 : tensor<11008x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<11008x4096xf32>
    %4154 = "tosa.reshape"(%4151) {new_shape = array<i64: 80, 11008>} : (tensor<1x80x11008xf32>) -> tensor<80x11008xf32>
    %cst_1450 = arith.constant dense<0.000000e+00> : tensor<80x4096xf32>
    %4155 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4154, %4153 : tensor<80x11008xf32>, tensor<11008x4096xf32>) outs(%cst_1450 : tensor<80x4096xf32>) -> tensor<80x4096xf32>
    %4156 = "tosa.reshape"(%4155) {new_shape = array<i64: 1, 80, 4096>} : (tensor<80x4096xf32>) -> tensor<1x80x4096xf32>
    %4157 = "tosa.add"(%4129, %4156) : (tensor<1x80x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %4158 = tensor.empty() : tensor<1x80x4096xf32>
    %c2_i32_1451 = arith.constant 2 : i32
    %4159 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4157 : tensor<1x80x4096xf32>) outs(%4158 : tensor<1x80x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4172 = math.fpowi %in, %c2_i32_1451 : f32, i32
      linalg.yield %4172 : f32
    } -> tensor<1x80x4096xf32>
    %cst_1452 = arith.constant dense<0.000000e+00> : tensor<1x80x1xf32>
    %4160 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%4159 : tensor<1x80x4096xf32>) outs(%cst_1452 : tensor<1x80x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1454 = arith.constant 4.096000e+03 : f32
      %4172 = arith.divf %in, %cst_1454 : f32
      %4173 = arith.addf %4172, %out : f32
      linalg.yield %4173 : f32
    } -> tensor<1x80x1xf32>
    %4161 = "tosa.const"() {value = dense<9.99999997E-7> : tensor<1x80x1xf32>} : () -> tensor<1x80x1xf32>
    %4162 = "tosa.add"(%4160, %4161) : (tensor<1x80x1xf32>, tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %4163 = "tosa.rsqrt"(%4162) : (tensor<1x80x1xf32>) -> tensor<1x80x1xf32>
    %4164 = "tosa.mul"(%4157, %4163) {shift = 0 : i32} : (tensor<1x80x4096xf32>, tensor<1x80x1xf32>) -> tensor<1x80x4096xf32>
    %4165 = "tosa.reshape"(%extracted_slice_63) {new_shape = array<i64: 1, 1, 4096>} : (tensor<4096xf32>) -> tensor<1x1x4096xf32>
    %4166 = "tosa.mul"(%4165, %4164) {shift = 0 : i32} : (tensor<1x1x4096xf32>, tensor<1x80x4096xf32>) -> tensor<1x80x4096xf32>
    %4167 = tensor.empty() : tensor<4096x32000xf32>
    %4168 = linalg.generic {indexing_maps = [#map3, #map8], iterator_types = ["parallel", "parallel"]} ins(%expanded_514 : tensor<32000x4096xf32>) outs(%4167 : tensor<4096x32000xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4096x32000xf32>
    %4169 = "tosa.reshape"(%4166) {new_shape = array<i64: 80, 4096>} : (tensor<1x80x4096xf32>) -> tensor<80x4096xf32>
    %cst_1453 = arith.constant dense<0.000000e+00> : tensor<80x32000xf32>
    %4170 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%4169, %4168 : tensor<80x4096xf32>, tensor<4096x32000xf32>) outs(%cst_1453 : tensor<80x32000xf32>) -> tensor<80x32000xf32>
    %4171 = "tosa.reshape"(%4170) {new_shape = array<i64: 1, 80, 32000>} : (tensor<80x32000xf32>) -> tensor<1x80x32000xf32>
    return %4171 : tensor<1x80x32000xf32>
  }
}
