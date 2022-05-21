memref.global "private" @gv : memref<5xf32> = dense<[0., 1., 2., 3., 4.]>

func.func @main() {
  %mem0 = memref.get_global @gv : memref<5xf32>
  %sum_0 = arith.constant 0.0 : f32 
  %result = arith.constant 5.5 : f32
  %lower = arith.constant 0 : index
  %upper = arith.constant 5 : index 
  %step  = arith.constant 1 : index 
  %sum = scf.for %i = %lower to %upper step  %step iter_args(%sum_iter = %sum_0 ) -> f32 {
    %t = memref.load %mem0[%i] : memref<5xf32>
    %sum_next = arith.addf %sum_iter, %t : f32
    scf.yield %sum_next : f32
  }   
  vector.print %sum : f32    
  func.return 

}
