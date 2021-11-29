# Copyright (c) 2021 buddy-compiler Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import argparse

import cv2
import tvm
from tvm import te
import numpy as np

IMAMGE_PATH = '../images/YuTu2048.png'

TILED_HEIGHT, TILED_WEIGHT = 3, 3  # Tile sizes for height and weight

sobel_3x3 = np.array([[1, 0, -1],
                      [2, 0, -2], 
                      [1, 0, -1]], dtype='float32')

sobel_5x5 = np.array([[2, 1, 0, -1, -2],
                      [3, 2, 0, -2, -3],  
                      [4, 3, 0, -3, -4], 
                      [3, 2, 0, -2, -3], 
                      [2, 1, 0, -1, -2]], dtype='float32')

sobel_7x7 = np.array([[3, 2, 1, 0, -1, -2, -3],
                      [4, 3, 2, 0, -2, -3, -4],  
                      [5, 4, 3, 0, -3, -4, -5], 
                      [6, 5, 4, 0, -4, -5, -6], 
                      [5, 4, 3, 0, -3, -4, -5], 
                      [4, 3, 2, 0, -2, -3, -4], 
                      [3, 2, 1, 0, -1, -2, -3]], dtype='float32')

sobel_9x9 = np.array([[4, 3, 2, 1, 0, -1, -2, -3, -4],
                      [5, 4, 3, 2, 0, -2, -3, -4, -5], 
                      [6, 5, 4, 3, 0, -3, -4, -5, -6], 
                      [7, 6, 5, 4, 0, -4, -5, -6, -7], 
                      [8, 7, 6, 5, 0, -5, -6, -7, -8], 
                      [7, 6, 5, 4, 0, -4, -5, -6, -7], 
                      [6, 5, 4, 3, 0, -3, -4, -5, -6], 
                      [5, 4, 3, 2, 0, -2, -3, -4, -5], 
                      [4, 3, 2, 1, 0, -1, -2, -3, -4]], dtype='float32')

sobel_3x3_filter = sobel_3x3.reshape((1, 1, 3, 3))
sobel_5x5_filter = sobel_5x5.reshape((1, 1, 5, 5))
sobel_7x7_filter = sobel_7x7.reshape((1, 1, 7, 7))
sobel_9x9_filter = sobel_9x9.reshape((1, 1, 9, 9))

def padding(X, ph, pw, val=0):
  """Pad X with the given value in 2-D

  ph, pw : height and width padding
  val : padding value, default 0
  """
  assert len(X.shape) >= 2
  nh, nw = X.shape[-2], X.shape[-1]
  return te.compute(
          (*X.shape[0:-2], nh+ph*2, nw+pw*2),
          lambda *i: te.if_then_else(
              te.any(i[-2]<ph, i[-2]>=nh+ph, i[-1]<pw, i[-1]>=nw+pw),
              val, X[i[:-2]+(i[-2]-ph, i[-1]-pw)]),
          name='PaddedX')

def conv_out_size(n, k, p, s):
  """Compute the output size by given input size n (width or height),
  kernel size k, padding p, and stride s
  Return output size (width or height)
  """
  print("parameters for computing out matrix", n, k, p, s)
  return (n - k + 2 * p) // s + 1

def conv(oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
  """Convolution

  oc, ic : output and input channels
  nh, nw : input width and height
  kh, kw : kernel width and height
  ph, pw : height and width padding sizes, default 0
  sh, sw : height and width strides, default 1
  """
  # reduction axes
  ric = te.reduce_axis((0, ic), name='ric')
  rkh = te.reduce_axis((0, kh), name='rkh')
  rkw = te.reduce_axis((0, kw), name='rkw')

  # output height and width
  oh = conv_out_size(nh, kh, ph, sh)
  ow = conv_out_size(nw, kw, pw, sw)

  # pad X and then compute Y
  X = te.placeholder((ic, nh, nw), name='X')
  K = te.placeholder((oc, ic, kh, kw), name='K')
  PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
  Y = te.compute(
      (oc, oh, ow),
      lambda c, i, j: te.sum(
          PaddedX[ric, i*sh+rkh, j*sw+rkw] * K[c, ric, rkh, rkw],
          axis=[ric, rkh, rkw]), name='Y')
  return X, K, Y, PaddedX

def get_conv_data(oc, ic, p=0, s=1, constructor=None):
  """Return random 3-D data tensor, 3-D kernel tenor and empty 3-D output 
  tensor with the shapes specified by input arguments.

  oc, ic : output and input channels
  p : padding size, default 0
  s : stride, default 1
  constructor : user-defined tensor constructor
  """
  img = cv2.imread(IMAMGE_PATH, cv2.IMREAD_GRAYSCALE)
  img = np.array(img, dtype='float32')
  n = img.shape[-1]
  data = img.reshape((ic, n, n))

  weight = sobel_3x3_filter
  k = weight.shape[-1]

  on = conv_out_size(n, k, p, s)
  out = np.empty((oc, on, on), dtype='float32')
  if constructor:
    data, weight, out = (constructor(x) for x in [data, weight, out])
  return data, weight, out

def test_conv2d_with_kernel(kernel, target):
  """Prepare tvm module with customized kernels 
  and tests its performance

  kernel : customized kernel
  target : customized target
  """
  oc, ic, p, s = 1, 1, 1, 1
  device = tvm.device(target, 0)
  data, weight, out = get_conv_data(oc, ic, p, s, lambda x: tvm.nd.array(x, device=device))
  n, k = data.shape[-1], weight.shape[-1]

  X, K, Y, _ = conv(oc, ic, n, n, k, k, p, p, s, s)
  sch = te.create_schedule(Y.op)
  mod = tvm.build(sch, [X, K, Y], target)
  print(tvm.lower(sch, [X, K, Y], simple_mode=True))

  start = time.time()
  mod(data, weight, out)
  end = time.time()
  print("evaluation time: %s" % ((end - start)))
  out = out.asnumpy().squeeze()
  return out

def cached_block(oc, ic, n, k, p, s):
  """Optimization recommended from TVM examples.
  Using several technics, including reorder, vectorize, 
  unroll and compute_at taken from TVM cpu optimization tutorial.

  oc, ic : output and input channels
  p : padding size, default 0
  s : stride, default 1
  k : kernel size, default 3
  """
  X, K, Y, PaddedX = conv(oc, ic, n, n, k, k, p, p, s, s)
  sch = te.create_schedule(Y.op)
  CachedY = sch.cache_write(Y, 'local')

  # Compute the output block for every output channel in parallel
  oc, h, w = Y.op.axis
  ho, wo, hi, wi = sch[Y].tile(h, w, TILED_HEIGHT, TILED_WEIGHT)
  ochw = sch[Y].fuse(oc, ho, wo)
  sch[Y].parallel(ochw)

  # Cache the output block, and move the inner height and width axes
  # to innermost, so we can vectorize and unroll them
  sch[CachedY].compute_at(sch[Y], ochw)
  _,  ch, cw = CachedY.op.axis
  ric, rkh, rkw = CachedY.op.reduce_axis
  sch[CachedY].reorder(ric, rkh, rkw, ch, cw)
  sch[CachedY].vectorize(cw)
  sch[CachedY].unroll(ch)

  # Schedule the padding by adding thread-level parallelism
  if PaddedX != X:
    sch[PaddedX].parallel(PaddedX.op.axis[0])
  return sch, (X, K, Y)

def test_cache_optimization():
  """Using official strategy taken from TVM cpu optimization
  example to optimize tvm module with customized kernels 
  and tests its performance.
  """
  oc, ic, p, s = 1, 1, 1, 1
  print('======================================================')
  sch, args = cached_block(oc, ic, 2048, 3, p, s)
  mod = tvm.build(sch, args)
  print(tvm.lower(sch, args, simple_mode=True))

  data, weight, out = get_conv_data(oc, ic, p, s, tvm.nd.array)
  n, k = data.shape[-1], weight.shape[-1]

  start = time.time()
  mod(data, weight, out)
  end = time.time()
  print("evaluation time: %s" % ((end - start)))
  out = out.asnumpy().squeeze()
  return out

def main(args):
  kernel_size, target = args.size, args.target
  print("processing kernel size: %s, with target: %s" % (kernel_size, target))
  if kernel_size == 3:
    kernel_chosen = sobel_3x3_filter
  elif kernel_size == 5:
    kernel_chosen = sobel_5x5_filter
  elif kernel_size == 7:
    kernel_chosen = sobel_7x7_filter
  elif kernel_size == 9:
    kernel_chosen = sobel_9x9_filter
  else:
    raise IndexError("only support kernel size in (3, 5, 7, 9)")
  edge_detect = test_conv2d_with_kernel(kernel_chosen, target)
  cv2.imwrite("./tvm-conv2d.png", edge_detect)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='specify kernel size and target')
  parser.add_argument('--size', metavar='N', type=int, default=3, help='an integer describe kernel size')
  parser.add_argument('-t', '--target', default='llvm -mcpu=skylake-avx512' ,help="target flag")
  args = parser.parse_args()
  # test for optimized version 
  # test_cache_optimization()
  main(args)
