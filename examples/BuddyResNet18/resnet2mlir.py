#----------------------------------完整模型结构----------------------------------
import os
from pathlib import Path
import numpy as np
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from torchvision.models import resnet18

# Load the ResNet-18 model
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(torch.load("resnet18.pth", map_location=torch.device('cpu')))
model = model.eval()

# Initialize Dynamo Compiler with specific configurations as an importer.
dynamo_compiler = DynamoCompiler(
    primary_registry=tosa.ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

for layer in model.modules():
    if isinstance(layer, torch.nn.BatchNorm2d):
        if hasattr(layer, "num_batches_tracked"):
            del layer.num_batches_tracked

# Example input for the ResNet model (3 channels for RGB images)
data = torch.randn([1, 3, 224, 224])

# Import the model into MLIR module and parameters.
with torch.no_grad():
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
params = dynamo_compiler.imported_params[graph]
pattern_list = [simply_fuse]
graphs[0].fuse_ops(pattern_list)
driver = GraphDriver(graphs[0])
driver.subgraphs[0].lower_to_top_level_ir()
path_prefix = os.path.dirname(os.path.abspath(__file__))

# Save the MLIR representation of the model.
with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
    print(driver.subgraphs[0]._imported_module, file=module_file)
with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
    print(driver.construct_main_graph(True), file=module_file)

# Debug step: Check the parameters before saving them
print("Checking parameters before saving to .data file...")

# 过滤掉 'num_batches_tracked' 参数
# filtered_params = [
#     param for name, param in model.state_dict().items() if 'num_batches_tracked' not in name
# ]

params = dynamo_compiler.imported_params[graph]
current_path = os.path.dirname(os.path.abspath(__file__))


float32_param = np.concatenate(
    [
        param.detach().numpy().reshape([-1])
        for param in params
        if param.dtype == torch.float32
    ]
)
float32_param.tofile(Path(current_path) / "arg0_resnet18.data")

# Print the shape and some initial values of each parameter tensor
# for idx, param in enumerate(params):
#     print(f"Parameter {idx}: shape = {param.shape}")
#     print(f"First 10 values of parameter {idx}: {param.flatten()[:10].tolist()}")

# 保存参数并去除num_batches_tracked
# float32_param = np.concatenate(
#     [param.detach().to(torch.float32).numpy().reshape([-1]) for param in filtered_params]
# )

# # Save to .data file
# float32_param.tofile(Path(path_prefix) / "arg0_resnet18.data")
# print("Parameters successfully saved to arg0_resnet18.data")

# # Debug step: Check concatenated parameter array
# print("Checking concatenated parameter array...")
# print(f"Total number of parameters (after concatenation): {len(float32_param)}")


data = np.fromfile('arg0_resnet18.data', dtype=np.float32)

# 打印数据的大小和前 20 个参数
print(f"Total number of parameters: {len(data)}")
print("First 20 parameters:")
print(data[:20])



# ------------------------------------只保留第一个卷积层--------------------------

# import os
# from pathlib import Path
# import numpy as np
# import torch
# from torch._inductor.decomposition import decompositions as inductor_decomp
# from buddy.compiler.frontend import DynamoCompiler
# from buddy.compiler.graph import GraphDriver
# from buddy.compiler.graph.transform import simply_fuse
# from buddy.compiler.ops import tosa
# from torchvision.models import resnet18

# # 截断 ResNet-18 模型，只保留第一个卷积层
# class TruncatedResNet18(torch.nn.Module):
#     def __init__(self, original_model):
#         super(TruncatedResNet18, self).__init__()
#         self.conv1 = original_model.conv1  # 仅保留第一个卷积层
#         self.bn1 = original_model.bn1

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         return x

# # 加载 ResNet-18 模型并截断
# model = resnet18(weights=None, num_classes=10)
# model.load_state_dict(torch.load("resnet18.pth", map_location=torch.device('cpu')))
# truncated_model = TruncatedResNet18(model).eval()

# # 初始化 Dynamo Compiler 作为导入器
# dynamo_compiler = DynamoCompiler(
#     primary_registry=tosa.ops_registry,
#     aot_autograd_decomposition=inductor_decomp,
# )

# # 示例输入数据
# data = torch.randn([1, 3, 224, 224])

# # 导入模型的第一个卷积层为 MLIR 模块并获取参数
# with torch.no_grad():
#     graphs = dynamo_compiler.importer(truncated_model, data)

# # 确保只导入了一个图并提取该图和参数
# assert len(graphs) == 1
# graph = graphs[0]
# params = dynamo_compiler.imported_params[graph]

# # 应用融合优化模式
# pattern_list = [simply_fuse]
# graphs[0].fuse_ops(pattern_list)

# # 创建 GraphDriver 进行 IR 转换
# driver = GraphDriver(graphs[0])
# driver.subgraphs[0].lower_to_top_level_ir()
# path_prefix = os.path.dirname(os.path.abspath(__file__))

# # 保存第一个卷积层的 MLIR 表示
# with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
#     print(driver.subgraphs[0]._imported_module, file=module_file)
# with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
#     print(driver.construct_main_graph(True), file=module_file)

# # 检查和保存第一个卷积层的参数
# print("Checking parameters before saving to .data file...")
# filtered_params = [
#     param for name, param in truncated_model.state_dict().items()
#     if 'num_batches_tracked' not in name and 'fc' not in name
# ]
# float32_param = np.concatenate(
#     [param.detach().to(torch.float32).numpy().reshape([-1]) for param in filtered_params]
# )

# # 将参数保存到 .data 文件
# float32_param.tofile(Path(path_prefix) / "arg0_resnet18.data")
# print("Parameters successfully saved to conv1_arg0_resnet18.data")

# # 打印参数的前 20 个值进行验证
# print(f"Total number of parameters (after concatenation): {len(float32_param)}")
# print(f"First 20 values of concatenated parameters: {float32_param[:20]}")





#---------------------------------保留除最后一个全连接层的所有层------------------------


# import os
# from pathlib import Path
# import numpy as np
# import torch
# from torch._inductor.decomposition import decompositions as inductor_decomp
# from buddy.compiler.frontend import DynamoCompiler
# from buddy.compiler.graph import GraphDriver
# from buddy.compiler.graph.transform import simply_fuse
# from buddy.compiler.ops import tosa
# from torchvision.models import resnet18

# 定义截断的 ResNet-18 模型，只保留到全连接层（fc）之前
# class TruncatedResNet18(torch.nn.Module):
#     def __init__(self, original_model):
#         super(TruncatedResNet18, self).__init__()
#         # 复制 ResNet-18 中的所有层，直到全连接层 fc 之前
#         self.features = torch.nn.Sequential(
#             original_model.conv1,
#             # original_model.bn1,
#             # original_model.relu,
#             # original_model.maxpool,
#             # original_model.layer1,
#             # original_model.layer2,
#             # original_model.layer3,
#             # original_model.layer4,
#             # original_model.avgpool
#         )

#     def forward(self, x):
#         x = self.features(x)
#         # 展平输出，使其适合全连接层的输入
#         x = torch.flatten(x, 1)
#         return x
    
    
# class TruncatedResNet18(torch.nn.Module):
#     def __init__(self, original_model):
#         super(TruncatedResNet18, self).__init__()
#         self.features = torch.nn.Sequential(
#             original_model.conv1,
#             original_model.bn1,
#             # original_model.relu,
#             # original_model.maxpool,
#             # original_model.layer1,
#             # original_model.layer2,
#             # original_model.layer3,
#             # original_model.layer4,
#             # original_model.avgpool
#         )


#     def forward(self, x):
#         x= self.features(x)
#         return x


# # 加载 ResNet-18 模型并截断到全连接层之前
# model = resnet18(weights=None, num_classes=10)
# model.load_state_dict(torch.load("resnet18.pth", map_location=torch.device('cpu')))
# truncated_model = TruncatedResNet18(model).eval()

# # 初始化 Dynamo Compiler 作为导入器
# dynamo_compiler = DynamoCompiler(
#     primary_registry=tosa.ops_registry,
#     aot_autograd_decomposition=inductor_decomp,
# )

# # 示例输入数据
# data = torch.randn([1, 3, 224, 224])

# # 导入截断模型为 MLIR 模块并获取参数
# with torch.no_grad():
#     graphs = dynamo_compiler.importer(truncated_model, data)

# # 确保只导入了一个图并提取该图和参数
# assert len(graphs) == 1
# graph = graphs[0]
# params = dynamo_compiler.imported_params[graph]

# # 应用融合优化模式
# pattern_list = [simply_fuse]
# graphs[0].fuse_ops(pattern_list)

# # 创建 GraphDriver 进行 IR 转换
# driver = GraphDriver(graphs[0])
# driver.subgraphs[0].lower_to_top_level_ir()
# path_prefix = os.path.dirname(os.path.abspath(__file__))

# # 保存截断模型的 MLIR 表示
# with open(os.path.join(path_prefix, "subgraph0.mlir"), "w") as module_file:
#     print(driver.subgraphs[0]._imported_module, file=module_file)
# with open(os.path.join(path_prefix, "forward.mlir"), "w") as module_file:
#     print(driver.construct_main_graph(True), file=module_file)

# # 检查并保存截断模型的参数
# print("Checking parameters before saving to .data file...")

# # 过滤掉 'num_batches_tracked' 参数
# filtered_params = [
#     param for name, param in model.state_dict().items()
#     if 'num_batches_tracked' not in name and 'fc' not in name
# ]

# # 将参数转换为 float32 并拼接成一个数组
# float32_param = np.concatenate(
#     [param.detach().to(torch.float32).numpy().reshape([-1]) for param in filtered_params]
# )

# # 保存为 .data 文件
# float32_param.tofile(Path(path_prefix) / "arg0_resnet18.data")
# print("Filtered parameters successfully saved to arg0_resnet18_truncated.data")

# # 输出前 20 个参数的值进行验证
# print("Checking concatenated parameter array...")
# print(f"Total number of parameters (after concatenation): {len(float32_param)}")
# print(f"First 20 values of concatenated parameters: {float32_param[:20]}")

# # 检查导出的参数文件内容
# data = np.fromfile('arg0_resnet18.data', dtype=np.float32)
# print(f"Total number of parameters in file: {len(data)}")
# print("First 20 parameters in file:")
# print(data[:20])
