# ===- pytorch-lenet-inference.py ----------------------------------------------
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
# LeNet inference with PyTorch runtime.
#
# ===---------------------------------------------------------------------------

import torch
from torchvision import transforms
from PIL import Image

from model import LeNet

# Load model
model = LeNet()
torch.load("./lenet-model.pth")
# Set the model to evaluation mode
model.eval()

# Prepare image and convert to grayscale
image_path = "./images/3.png"
image = Image.open(image_path).convert("L")

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
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
# Add batch dimension: [CHW] -> [NCHW]
image = transform(image).unsqueeze(0)

# Perform inference
# No gradient tracking in this block
with torch.no_grad():
    output = model(image)
    prediction = output.argmax()

print(f"Classification: {prediction.item()}")
