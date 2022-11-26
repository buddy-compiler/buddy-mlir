// `vector.bitcast` creates an vector from one type to another type, with the 
// exactly same bit pattern. 

func.func @main() -> i32 {
  %v0 = arith.constant dense<[10, 20, 56, 90, 12, 90]> : vector<6xi32>
  vector.print %v0 : vector<6xi32>

  // bitcast can change the element type and dimension
  %v1 = vector.bitcast %v0 : vector<6xi32> to vector<3xi64>
  vector.print %v1 : vector<3xi64>

  // it can even change element type from integer to float
  // note that it will preserve bit pattern instead of value
  %v2 = vector.bitcast %v0 : vector<6xi32> to vector<6xf32>
  vector.print %v2 : vector<6xf32>

  // cast it back and it will be the same vector with exactly
  // every bit same as %v0 
  %v3 = vector.bitcast %v2 : vector<6xf32> to vector<6xi32>
  vector.print %v3 : vector<6xi32>

  // bitcast could only be used between vector types with
  // same total length in byte, like 8xi32 <-> 4xf64

  // error: 'vector.bitcast' op source/result bitwidth of the minor 1-D vectors must be equal
  // %v4 = vector.bitcast %v0 : vector<6xi32> to vector<4xi64> 


  // Because scalable vector is platform specify, vector dialect could not
  // lowering/translating them well, so we just assume that we have one:
  //                %v5 : vector<[4]xi32>
  // That's also why we have to comment out the operations below, even if 
  // they are valid usage.

  // bitcast will also accept scalable dimensions
  // %v6 = vector.bitcast %v5 : vector<[4]xi32> to vector<[2]xi64>
  // vector.print %v6 : vector<[2]xi64>

  // %v7 = vector.bitcast %v5 : vector<[4]xi32> to vector<[8]xi16>
  // vector.print %v7 : vector<[8]xi16>

  // bitcast operations of scalable dimensions must respect the bitwidth
  // restriction as same with fix length dimensions 

  // error: 'vector.bitcast' op source/result bitwidth of the minor 1-D vectors must be equal
  // %v8 = vector.bitcast %v5 : vector<[4]xi32> to vector<[3]xi64>

  // bitcast operations of scalable dimensions should ALWAYS meet the 
  // bitwidth restriction, not just POSSIBLE to meet it.  

  // error: 'vector.bitcast' op source/result bitwidth of the minor 1-D vectors must be equal
  // %v9 = vector.bitcast %v5 : vector<[4]xi32> to vector<[4]xi64>

  %ret = arith.constant 0 : i32
  return %ret : i32
}
