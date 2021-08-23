import onnx
from onnx import numpy_helper, helper, TensorProto
import numpy as np

# Filter
sobel_3x3 = np.array([[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]], dtype='float32')
w = sobel_3x3.reshape((1, 1, 3, 3))

# Input
x = np.random.rand(1, 1, 2048, 2048).astype('float32')

# Initializer of the weight
initializer_w = numpy_helper.from_array(w, 'w')

tensor_w = helper.make_tensor_value_info('w', TensorProto.FLOAT, [1, 1, 3, 3])
tensor_x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2048, 2048])
tensor_y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 2046, 2046])

# Create a node
node_def = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'w'],
    outputs=['y'],
    kernel_shape=[3, 3]
)

# Create the graph
graph_def = helper.make_graph(
    [node_def],
    'conv_3x3',
    [tensor_x],
    [tensor_y],
    [initializer_w]
)

# Create the model
model_def = helper.make_model(graph_def,
        producer_name='python_script',
        ir_version=6
)
model_def.opset_import[0].version = 10

# Check the model
onnx.checker.check_model(model_def)

# Save the model
onnx.save(model_def, 'conv_3x3.onnx')
