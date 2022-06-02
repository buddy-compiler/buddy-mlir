func.func @main() {
  %vector_1 = arith.constant dense<[[12.,13.,24.,67.,75.],
                                      [23.,25.,45.,67.,78.],
                                      [67.,90.,78.,90.,91.]]> : vector<3x5xf32>

  %vector_1_trans = vector.transpose %vector_1, [1,0] : vector<3x5xf32> to vector<5x3xf32> // Exchanges dims 0 and 1
  vector.print %vector_1_trans : vector<5x3xf32>

  %vector_2 = arith.constant dense<[[12.,23.,45.],
                                     [23.,45.,67.]]> : vector<2x3xf32>

  %vector_2_trans = vector.transpose %vector_2, [0,1] : vector<2x3xf32> to vector<2x3xf32> // Keeps the same vector dimensions
  vector.print %vector_2_trans : vector<2x3xf32>
    
  %vector_3 = arith.constant dense<[[[45.,56.,78.,90.,12.],[23.,67.,90.,45.,54.],[23.,89.,100.,101.,114.],[123.,245.,67.,78.,90.]],
                                      [[451.,50.,79.,100.,12.],[29.,60.,91.,47.,50.],[28.,88.,109.,135.,104.],[123.,240.,64.,79.,99.]],
                                      [[45.,59.,77.,99.,121.],[25.,69.,99.,47.,58.],[13.,79.,101.,102.,115.],[124.,248.,671.,90.,234.]]]> : vector<3x4x5xf32>

  %vector_3_trans = vector.transpose %vector_3, [1,2,0] : vector<3x4x5xf32> to vector<4x5x3xf32>  // Changes all three dimensions
  vector.print %vector_3_trans : vector<4x5x3xf32>

  return                                  
}
