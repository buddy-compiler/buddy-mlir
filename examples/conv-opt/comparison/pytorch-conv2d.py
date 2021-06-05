import torch
import numpy as np
import cv2
import time
from torch.autograd import Variable
import torch.nn.functional as F

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
 
def test_conv2d(img, kernel):
  weight = Variable(torch.from_numpy(kernel))
  start = time.time()
  edge_detect = F.conv2d(Variable(img), weight)
  end = time.time()
  print(end - start)
  edge_detect = edge_detect.squeeze().detach().numpy()
  return edge_detect
 
def main():
  img = cv2.imread('../images/YuTu2048.png',cv2.IMREAD_GRAYSCALE)
  # Convert the image to numpy array.
  img = np.array(img, dtype='float32')
  # Convert the numpy array to torch tensor.
  img = torch.from_numpy(img.reshape((1, 1, img.shape[0], img.shape[1])))
  '''
  Perform the edget detection.
  Uncomment to use the corresponding size kernel for testing.
  Note that only one kernel size is used for testing at a time.
  '''
  edge_detect = test_conv2d(img, sobel_3x3_filter)
  # edge_detect = test_conv2d(img, sobel_5x5_filter)
  # edge_detect = test_conv2d(img, sobel_7x7_filter)
  # edge_detect = test_conv2d(img, sobel_9x9_filter)
  cv2.imwrite("./pytorch-conv2d.png", edge_detect)
 
if __name__ == "__main__":
  main()
