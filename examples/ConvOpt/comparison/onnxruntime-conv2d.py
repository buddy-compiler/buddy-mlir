import numpy as np
import cv2
import onnxruntime
import time

def test_conv2d(img, filter_size):
  start = time.time()
  # Load the model
  model_path = f'conv_{filter_size}x{filter_size}.onnx'
  ort_session = onnxruntime.InferenceSession(model_path)
  # Run inference
  ort_inputs = {ort_session.get_inputs()[0].name: img}
  ort_outs = ort_session.run(None, ort_inputs)
  edge_detect = ort_outs[0]
  edge_detect = edge_detect.squeeze()
  end = time.time()
  print(f'conv {filter_size}x{filter_size} : {end - start}')
  return edge_detect

def main():
  img = cv2.imread('../images/YuTu2048.png',cv2.IMREAD_GRAYSCALE)
  # Convert the image to numpy array.
  img = np.array(img, dtype='float32')
  img = img.reshape((1, 1, img.shape[0], img.shape[1]))
  '''
  Perform the edget detection.
  '''
  for i in range(3, 10, 2):
    edge_detect = test_conv2d(img, i)
    cv2.imwrite(f'./onnxruntime-conv2d_{i}.png', edge_detect)

if __name__ == "__main__":
  main()
