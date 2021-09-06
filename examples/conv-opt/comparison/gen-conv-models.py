import onnx
from onnx import numpy_helper
import numpy as np

# Filter
sobel = {
  3: np.array([[1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]], dtype='float32'),
  5: np.array([[2, 1, 0, -1, -2],
    [3, 2, 0, -2, -3],
    [4, 3, 0, -3, -4],
    [3, 2, 0, -2, -3],
    [2, 1, 0, -1, -2]], dtype='float32'),
  7: np.array([[3, 2, 1, 0, -1, -2, -3],
    [4, 3, 2, 0, -2, -3, -4],
    [5, 4, 3, 0, -3, -4, -5],
    [6, 5, 4, 0, -4, -5, -6],
    [5, 4, 3, 0, -3, -4, -5],
    [4, 3, 2, 0, -2, -3, -4],
    [3, 2, 1, 0, -1, -2, -3]], dtype='float32'),
  9: np.array([[4, 3, 2, 1, 0, -1, -2, -3, -4],
    [5, 4, 3, 2, 0, -2, -3, -4, -5],
    [6, 5, 4, 3, 0, -3, -4, -5, -6],
    [7, 6, 5, 4, 0, -4, -5, -6, -7],
    [8, 7, 6, 5, 0, -5, -6, -7, -8],
    [7, 6, 5, 4, 0, -4, -5, -6, -7],
    [6, 5, 4, 3, 0, -3, -4, -5, -6],
    [5, 4, 3, 2, 0, -2, -3, -4, -5],
    [4, 3, 2, 1, 0, -1, -2, -3, -4]], dtype='float32')
}

def get_output_shape(i):
  if i == 3:
    return [1, 1, 2046, 2046]
  elif i == 5:
    return [1, 1, 2044, 2044]
  elif i == 7:
    return [1, 1, 2042, 2042]
  elif i == 9:
    return [1, 1, 2040, 2040]

def main():
  for i in range(3, 10, 2):
    # Filter
    w = sobel[i].reshape((1, 1, i, i))

    # Input
    x = np.random.rand(1, 1, 2048, 2048).astype('float32')

    # Initializer of the weight
    initializer_w = numpy_helper.from_array(w, 'w')

    tensor_w = onnx.helper.make_tensor_value_info('w', onnx.TensorProto.FLOAT, [1, 1, i, i])
    tensor_x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 1, 2048, 2048])
    tensor_y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, get_output_shape(i))

    # Create a node
    node_def = onnx.helper.make_node(
      'Conv',
      inputs=['x', 'w'],
      outputs=['y'],
      kernel_shape=[i, i]
    )

    # Create the graph
    graph_def = onnx.helper.make_graph(
      [node_def],
      f'conv_{i}x{i}',
      [tensor_x],
      [tensor_y],
      [initializer_w]
    )

    # Create the model
    model_def = onnx.helper.make_model(graph_def,
      producer_name='python_script',
      ir_version=6
    )
    model_def.opset_import[0].version = 10

    # Check the model
    onnx.checker.check_model(model_def)

    # Save the model
    onnx.save(model_def, f'conv_{i}x{i}.onnx')

if __name__ == "__main__":
  main()
