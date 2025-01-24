# ===- BuddyJIT.py -------------------------------------------------------------
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
#
# ===---------------------------------------------------------------------------
#
# This is the JIT mode compilation and execution of the LeNet model.
#
# ===---------------------------------------------------------------------------
import os

import torch

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from model import LeNet

from PIL import Image
import torchvision.transforms as transforms

# Retrieve the LeNet model path.
model_path = os.path.dirname(os.path.abspath(__file__))

model = LeNet()
model = torch.load(model_path + "/lenet-model.pth", weights_only=False)
model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    compilation_mode="JIT",
)

image_path = model_path + "/1-28*28.png"

# Resize image to match the model's expected input dimensions
# Convert to tensor
#   - This conversion is achieved by dividing the original pixel values by 255.
#   - Before: An image with pixel values typically in the range [0, 255].
#   - After: A PyTorch tensor with the shape (C, H, W) and pixel values
#            normalized to [0.0, 1.0].
# Normalize
#   - This step normalizes each channel of the tensor to have a mean of 0.5 and
#     a standard deviation of 0.5.
#   - Before: A tensor with pixel values in the range [0.0, 1.0].
#   - After: A tensor with pixel values normalized to the range [-1.0, 1.0],
#            making the network training process more stable and faster by
#            standardizing the range of input values.
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

image = Image.open(image_path)
# Add batch dimension: [CHW] -> [NCHW]
input_tensor = transform(image).unsqueeze(0)

model_opt = torch.compile(model, backend=dynamo_compiler)
with torch.no_grad():
    print("Pytorch Output:", model(input_tensor))
    output = model_opt(input_tensor)
    print("Model Output:", output)

predicted_class = torch.argmax(output, dim=1).item()
print(f"Predicted Class: {predicted_class}")
