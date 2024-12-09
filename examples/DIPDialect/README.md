If you want to test the functionality of image

Please follow:
$ cd buddy-mlir/build
$ cmake -G Ninja .. -DBUDDY_EXAMPLES=ON 
$ ninja resize4D_nchw
$ cd bin
$ ./resize4D_nhwc ../../examples/images/YuTu.png result-dip-resize.bmp