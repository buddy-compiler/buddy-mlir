import tensorflow as tf
import numpy as np
import cv2
import time

sobel_3x3 = tf.constant([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]], dtype='float32')

sobel_5x5 = tf.constant([[2, 1, 0, -1, -2],
                         [3, 2, 0, -2, -3],  
                         [4, 3, 0, -3, -4], 
                         [3, 2, 0, -2, -3], 
                         [2, 1, 0, -1, -2]], dtype='float32')

sobel_7x7 = tf.constant([[3, 2, 1, 0, -1, -2, -3],
                         [4, 3, 2, 0, -2, -3, -4],  
                         [5, 4, 3, 0, -3, -4, -5], 
                         [6, 5, 4, 0, -4, -5, -6], 
                         [5, 4, 3, 0, -3, -4, -5], 
                         [4, 3, 2, 0, -2, -3, -4], 
                         [3, 2, 1, 0, -1, -2, -3]], dtype='float32')

sobel_9x9 = tf.constant([[4, 3, 2, 1, 0, -1, -2, -3, -4],
                         [5, 4, 3, 2, 0, -2, -3, -4, -5], 
                         [6, 5, 4, 3, 0, -3, -4, -5, -6], 
                         [7, 6, 5, 4, 0, -4, -5, -6, -7], 
                         [8, 7, 6, 5, 0, -5, -6, -7, -8], 
                         [7, 6, 5, 4, 0, -4, -5, -6, -7], 
                         [6, 5, 4, 3, 0, -3, -4, -5, -6], 
                         [5, 4, 3, 2, 0, -2, -3, -4, -5], 
                         [4, 3, 2, 1, 0, -1, -2, -3, -4]], dtype='float32')

sobel_3x3_filter = tf.reshape(sobel_3x3, [3, 3, 1, 1])
sobel_5x5_filter = tf.reshape(sobel_5x5, [5, 5, 1, 1])
sobel_7x7_filter = tf.reshape(sobel_7x7, [7, 7, 1, 1])
sobel_9x9_filter = tf.reshape(sobel_9x9, [9, 9, 1, 1])

def test_conv2d(img, kernel):
  start = time.time()
  output = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='VALID')
  end = time.time()
  print(end - start)
  output_array = np.asarray(output)
  cv2.imwrite("./tf-conv2d.png", output_array[0,:,:,0])

def main():
  img = cv2.imread('../images/YuTu2048.png',cv2.IMREAD_GRAYSCALE)
  img = tf.constant(img, tf.float32)
  img = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
  '''
  Perform the edget detection.
  Uncomment to use the corresponding size kernel for testing.
  Note that only one kernel size is used for testing at a time.
  '''
  test_conv2d(img, sobel_3x3_filter)
  # test_conv2d(img, sobel_5x5_filter)
  # test_conv2d(img, sobel_7x7_filter)
  # test_conv2d(img, sobel_9x9_filter)

if __name__ == "__main__":
  main()
