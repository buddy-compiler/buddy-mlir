# documentation: https://pytorch.org/hub/pytorch_vision_once_for_all/
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from torch._functorch.aot_autograd import aot_autograd_decompositions
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp

super_net_name = "ofa_supernet_mbv3_w10"
# other options:
#    ofa_supernet_resnet50 /
#    ofa_supernet_mbv3_w12 /
#    ofa_supernet_proxyless

model = torch.hub.load(
    "mit-han-lab/once-for-all", super_net_name, pretrained=True
).eval()

# Download an example image from pytorch website
import urllib.request

url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "dog.jpg",
)
urllib.request.urlretrieve(url, filename)


# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

input_image = Image.open(filename)
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(
    0
)  # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# ---------------- success ------------------
# import torch
# from torch._dynamo import optimize
# import torch._inductor.config
# torch._inductor.config.debug = True
# new_fn = optimize("inductor")(model)
# a = new_fn(input_batch)


# ----------------- FAILED ------------------

import torch._inductor.config

torch._inductor.config.debug = True
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=aot_autograd_decompositions,
    is_inference=True,
)


gm, params = dynamo_compiler.importer(model, input_batch)
"""
  File "/home/xxx/buddy-mlir/mlir/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 2562, in aot_wrapper_dedupe
    return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)
  File "/home/xxx/buddy-mlir/mlir/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 2749, in aot_wrapper_synthetic_base
    return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)
  File "/home/xxx/buddy-mlir/mlir/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 3598, in aot_dispatch_autograd
    compiled_fw_func = aot_config.fw_compiler(
  File "/home/xxx/buddy-mlir/build/python_packages/buddy/compiler/frontend.py", line 119, in _compiler
    self._imported_module = fx_importer.import_graph()
  File "/home/xxx/buddy-mlir/build/python_packages/buddy/compiler/frontend.py", line 256, in import_graph
    tensor_size += functools.reduce(
torch._dynamo.exc.BackendCompilerFailed: backend='_compile_fx' raised:
TypeError: reduce() of empty iterable with no initial value

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
"""
