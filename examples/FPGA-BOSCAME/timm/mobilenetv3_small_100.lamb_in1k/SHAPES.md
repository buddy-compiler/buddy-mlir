# MobileNetV3 Small 100：shape 提取

原始模型 shape 采用与 `../../qwen3-0.6b/` 相同的 `MODEL.md` + `model.json`
组织方式。本文件描述原始 FP32 模型的 shape 提取；后续 BN folding 和
residual add 实现见 [README.md](README.md)。

模型来源：[timm/mobilenetv3_small_100.lamb_in1k](https://hf-mirror.com/timm/mobilenetv3_small_100.lamb_in1k)，
锁定 revision `1824797e7887cbec1990e4adbd6675960a36c589`。
脚本校验 config 和 safetensors 的 SHA-256，并严格加载全部权重。
这里使用 timm 的模型定义与指定 checkpoint；仓库已有 `BuddyMobileNetV3`
示例使用 torchvision 的模型和另一份权重，不能作为本次提取的来源。

默认 profile 为 batch=1、NCHW `[1,3,224,224]`、CPU、FP32、eval。
配置的 `fixed_input_size=false`，其他 batch/分辨率可通过脚本重新提取。
输入采用 seed=0 的随机张量；shape 不依赖图片内容。预处理配置保存在
`model.json` 的 `official_config.pretrained_cfg` 中，但图片 resize、crop、
normalize 和输出 softmax 不在本次 forward 的算子范围内。

## 结果文件

| 文件 | 内容 |
| --- | --- |
| [MODEL.md](MODEL.md) | 主干 shape、全部卷积/全连接、完整 ATen 序列、全部参数与 buffer |
| [model.json](model.json) | 来源与版本、验证结果、229 次模块调用、191 次 ATen 调用、244 个 checkpoint 张量、矩阵映射 |
| [operator_shapes.json](operator_shapes.json) | 按算子名、全部参数及输出描述去重的 110 个签名，附出现次数和来源位置 |
| [tools/extract_shapes.py](tools/extract_shapes.py) | 下载、严格加载、实跑和生成报告的可复现脚本 |

提取采用 forward hooks 记录模块输入/输出，同时用 `TorchDispatchMode`
记录实际 ATen 调用，因此包含模块内部的 SE 归约、广播乘法及残差加法。
算子参数保留归约维度、keepdim、groups、stride、padding、dilation、
BN epsilon、dtype 和张量 stride。`modules` 包含容器和叶子模块，
不能把其调用次数当作内核数相加。

本次结果为 2,542,856 个参数；53 个卷积（其中 11 个 depthwise）、
1 个 Linear、34 次 BN、9 组 SE、6 次残差加法。
卷积与 Linear 合计 56,510,400 MAC，不含 BN、激活、归约等运算。
191 次 ATen 调用包含 34 次 `empty` 分配；110 个签名不是待实现的
FPGA 算术内核数量。相同 shape 的算子仍可能具有不同归约轴或参数。

卷积的逻辑矩阵形状遵循 Qwen3 的 `M,N,K` 顺序：
`A[M,K] × B[K,N] → C[M,N]`。对每个 group，
`M=B×Hout×Wout`、`N=Cout/groups`、`K=(Cin/groups)×Kh×Kw`。
这是数学展开，尚未决定 im2col 或数据布局。权重实际存储为 OIHW，
depthwise 的 groups 必须保留。

## 复现

在已有匹配依赖的 Python 环境中运行：

```bash
python examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/tools/extract_shapes.py
```

本次环境为 Python 3.12、torch 2.10.0+cpu、torchvision 0.25.0+cpu、
timm 1.0.26、safetensors 0.8.0。为复用本机已有 PyTorch 环境，
timm 单独安装在被 git 忽略的 `build/python` 中；本机复现命令为：

```bash
PYTHONPATH=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/build/python \
  /home/zhangwenji/triton-riscv/.venv/bin/python \
  examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/tools/extract_shapes.py
```

缺少权重时会从 hf-mirror 下载到 `build/checkpoint`；原始权重和本地依赖
由本目录 `.gitignore` 忽略。可用 `--endpoint https://huggingface.co`
切换下载源，仍校验同一 revision 的文件摘要。

其他尺寸请指定单独输出目录，避免覆盖默认结果：

```bash
python examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/tools/extract_shapes.py \
  --batch-size 2 --height 256 --width 256 \
  --output-dir /tmp/mobilenetv3-b2-256
```

脚本检查记录前后 logits 逐字节一致、输出有限且为 `[batch,1000]`，
并用卷积公式独立核对空间尺寸，交叉核对 hooks 与 ATen 的全部卷积。
这验证 shape 提取与执行一致性，不是 ImageNet 精度评估。
