import torch
import torchvision
from torch._inductor.compile_fx import compile_fx
from PIL import Image
import requests
from torchvision.models import resnet18
from torchvision import transforms
from typing import List


def load_resnet_model() -> torchvision.models.ResNet:
    model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()

    return model


def preprocess_image(url: str) -> torch.Tensor:
    image = Image.open(requests.get(url, stream=True).raw)
    preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocessor(image)


IMAGE_URL = "https://housing.com/news/wp-content/uploads/2023/07/Cute-dog-breeds-that-make-the-best-pets-f-686x400.jpg"


def infer_by_cpu() -> torch.Tensor:
    model = load_resnet_model()
    image = preprocess_image(IMAGE_URL)
    input_batch = image.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)
    return probabilities


def triton_backend(graph_mode: torch.fx.GraphModule, example_input: List[torch.Tensor]) -> torch.nn.Module:
    graph_mode.graph.print_tabular()
    return compile_fx(graph_mode, example_input)


def triton_compiler(model: torch.nn.Module) -> torch.nn.Module:
    return torch._dynamo.optimize(backend=triton_backend)(model)


def infer_by_triton() -> torch.Tensor:
    model = load_resnet_model().to('cuda')
    triton_model = triton_compiler(model)
    image = preprocess_image(IMAGE_URL).to('cuda')
    input_batch = image.unsqueeze(0)
    with torch.no_grad():
        output = triton_model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)
    return probabilities


if __name__ == "__main__":
    cpu_probabilities = infer_by_cpu()
    triton_probabilities = infer_by_triton()

    print(f"Is CPU result equal to GPU: {torch.allclose(cpu_probabilities, triton_probabilities)}")
