# MobileNetV3 Small 100：FPGA 适配

已实现 10 类 Triton 算子、88 个静态 case：统一 inventory 自动生成 case、metadata 和独立 C oracle，支持 Host 验证与 NR ELF/BIN 构建。另有 shape 提取与 34 个 inference BatchNorm 的离线 folding 工具。

2026-10-08 提交前，88 个已有 Host 验证程序及 8 个 BN folding 回归测试全部通过。2026-10-06 的逐算子 FPGA 结果如下；这是 patch5/patch6 混合版本验证，尚未完成全部算子的同版 patch6 回归或完整模型上板。

| 算子 | case 数 | FPGA PASS 版本 | 说明 |
|---|---:|---|---|
| residual_add | 4 | nr-patch5 | [实现](triton/README.md) |
| relu | 11 | nr-patch5 | [实现](triton/RELU.md) |
| hardsigmoid | 7 | nr-patch5 | [实现](triton/HARDSIGMOID.md) |
| hardswish | 9 | nr-patch5 | [实现](triton/HARDSWISH.md) |
| se_mul | 7 | nr-patch5 | [实现](triton/SE_MUL.md) |
| mean_hw | 7 | nr-patch5 | [实现](triton/MEAN_HW.md) |
| linear | 1 | nr-patch6 | [实现](triton/LINEAR.md) |
| depthwise_conv2d | 9 | nr-patch6 | [实现](triton/DWCONV.md) |
| pointwise_conv2d | 32 | nr-patch6 | [实现](triton/PWCONV.md) |
| conv_stem | 1 | nr-patch5 | [实现](triton/STEM.md) |

Linear 已通过软件编译路径绕过 patch6 的 RVV 归约异常，原 kernel、C oracle 和容限保持不变；硬件归约问题本身仍未解决。

可随仓库查看的逐 case 结果和串口证据见 [FPGA 验证记录](validation/fpga-20261006/README.md)。早期算子文档中的 `NOT EXECUTED` 描述保留为初次构建时的历史记录。

当前完成离线 BN 权重折叠，固定 profile 为 batch=1、`[1,3,224,224]`、
NCHW、FP32、eval，输出 `[1,1000]`。原始权重、本地依赖和 shape 清单已合并到
本目录：权重在 `build/checkpoint/`，依赖在 `build/python/`，shape 清单见
[MODEL.md](MODEL.md)，提取说明见 [SHAPES.md](SHAPES.md)。
本目录是 FPGA 适配的实现目录。BN folding 是离线权重变换；
后续 residual add 实现在 `triton/` 下。没有更改 Qwen3 实现或公共编译/运行工具。

## 实现

- [`model/tools/fuse_batchnorm.py`](model/tools/fuse_batchnorm.py)：图驱动配对、
  权重折叠、完整模型比较、图与运行时审计、保存与重新加载融合模型。
- [`model/tools/test_fuse_batchnorm.py`](model/tools/test_fuse_batchnorm.py)：
  bias、分组/深度卷积、激活保留和不安全配对的回归测试。
- [`model/validation/bn-fusion/report.json`](model/validation/bn-fusion/report.json)：
  实测误差、全部 34 个融合位置、源文件摘要、输出文件摘要和验证结果。
- [`before.fx.txt`](model/validation/bn-fusion/before.fx.txt) /
  [`after.fx.txt`](model/validation/bn-fusion/after.fx.txt)：融合前后实际 FX 图。

工具要求先 `model.eval()`，按 FX 的真实生产者/消费者关系匹配
`Conv2d -> BatchNorm2d`，不依赖手写的 34 项层名清单。
只允许 Conv 输出被该 BN 独占，且 Conv/BN 均不重复调用。
工具深拷贝模型，原模型和原 checkpoint 保持原样。

对 OIHW 权重的输出通道 `o`：

```text
scale[o] = gamma[o] / sqrt(running_var[o] + eps)
W'[o,i,h,w] = W[o,i,h,w] * scale[o]
b'[o] = beta[o] + (b[o] - running_mean[o]) * scale[o]
```

原 Conv 没有 bias 时使用零向量；当前模型的 34 个配对均为这种情况。
保留原来的 groups、stride、padding、dilation 和权重布局。
非 affine BN 支持 `gamma=1, beta=0`，没有 running statistics 的 BN 拒绝融合。

本模型的 BN 都是 timm `BatchNormAct2d`，同时带有激活。
融合后原 BN 路径替换成保留 `drop`、`act` 子模块的 Sequential；
Hardswish/ReLU 保持原来的执行位置，BN 的缩放与偏置全部进入前置 Conv。
因此 `bn1.act` 等路径仍可能出现，但不再包含 BatchNorm 计算。

## Host 验证

在仓库根目录运行，以下命令对应当前机器已验证的环境：

```bash
PYTHONPATH=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/build/python \
  /home/zhangwenji/triton-riscv/.venv/bin/python \
  examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/model/tools/fuse_batchnorm.py

PYTHONPATH=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/build/python \
  /home/zhangwenji/triton-riscv/.venv/bin/python \
  examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/model/tools/test_fuse_batchnorm.py -v
```

依赖版本：Python 3.12、torch 2.10.0+cpu、timm 1.0.26、safetensors 0.8.0；
现有环境的 torchvision 为 0.25.0+cpu。有这些依赖的其他环境可直接用其
`python` 执行脚本，无需上述 `PYTHONPATH`。
工具从原始 checkpoint 缓存读取文件，缺失时从 hf-mirror 下载锁定 revision
`1824797e7887cbec1990e4adbd6675960a36c589`，并验证 SHA-256。

固定输入通过独立的 `torch.Generator().manual_seed(0)` 生成，避免模型初始化
消耗随机数后改变验收输入。参考是原始 PyTorch 完整模型的 FP32 输出。
这次是离线权重变换，没有 Triton 算子 case 或 C kernel oracle。

验收阈值为 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，
同时要求 Top-1 一致、输出有限且 shape 正确、34 个 BN 全部消失。
任一验证失败均以非零退出码结束，不输出 `COMPLETE`。

| 检查 | 实际结果 |
| --- | --- |
| 完整模型数值比较 | PASS |
| max_abs_error | 9.179115295410156e-06 |
| mean_abs_error | 1.6866140413185349e-06 |
| Top-1 | 858 → 858，一致（零起始类别索引） |
| BN 模块 / FX 节点 / ATen 调用 | 均为 34 → 0 |
| 其余计算调用计数 | 保持一致：53 Conv、19 Hardswish、14 ReLU、9 SE 乘法、6 残差 add 等 |
| 保存后严格加载并重新执行 | 与内存中的融合模型输出逐字节一致 |
| 边界回归测试 | 8 个测试全部通过 |

回归覆盖：Conv bias=None / 已有 bias、普通/分组/depthwise 卷积、
非方形输入和卷积核、padding/stride/dilation、非 affine BN、
负/零 gamma、timm 激活和 dropout 保留，以及训练态、无 running statistics、
Conv 输出分支和重复调用的拒绝行为。

## 全部融合位置

这 34 项是自动发现的图变换位置；完整模型数值验收使用一个固定 profile。
每个位置的卷积形状、groups、eps、原 bias 状态及保留激活见报告中的 `fusions`。

```text
conv_stem              -> bn1
blocks.0.0.conv_dw      -> blocks.0.0.bn1
blocks.0.0.conv_pw      -> blocks.0.0.bn2
blocks.1.0.conv_pw      -> blocks.1.0.bn1
blocks.1.0.conv_dw      -> blocks.1.0.bn2
blocks.1.0.conv_pwl     -> blocks.1.0.bn3
blocks.1.1.conv_pw      -> blocks.1.1.bn1
blocks.1.1.conv_dw      -> blocks.1.1.bn2
blocks.1.1.conv_pwl     -> blocks.1.1.bn3
blocks.2.0.conv_pw      -> blocks.2.0.bn1
blocks.2.0.conv_dw      -> blocks.2.0.bn2
blocks.2.0.conv_pwl     -> blocks.2.0.bn3
blocks.2.1.conv_pw      -> blocks.2.1.bn1
blocks.2.1.conv_dw      -> blocks.2.1.bn2
blocks.2.1.conv_pwl     -> blocks.2.1.bn3
blocks.2.2.conv_pw      -> blocks.2.2.bn1
blocks.2.2.conv_dw      -> blocks.2.2.bn2
blocks.2.2.conv_pwl     -> blocks.2.2.bn3
blocks.3.0.conv_pw      -> blocks.3.0.bn1
blocks.3.0.conv_dw      -> blocks.3.0.bn2
blocks.3.0.conv_pwl     -> blocks.3.0.bn3
blocks.3.1.conv_pw      -> blocks.3.1.bn1
blocks.3.1.conv_dw      -> blocks.3.1.bn2
blocks.3.1.conv_pwl     -> blocks.3.1.bn3
blocks.4.0.conv_pw      -> blocks.4.0.bn1
blocks.4.0.conv_dw      -> blocks.4.0.bn2
blocks.4.0.conv_pwl     -> blocks.4.0.bn3
blocks.4.1.conv_pw      -> blocks.4.1.bn1
blocks.4.1.conv_dw      -> blocks.4.1.bn2
blocks.4.1.conv_pwl     -> blocks.4.1.bn3
blocks.4.2.conv_pw      -> blocks.4.2.bn1
blocks.4.2.conv_dw      -> blocks.4.2.bn2
blocks.4.2.conv_pwl     -> blocks.4.2.bn3
blocks.5.0.conv         -> blocks.5.0.bn1
```

## 保存和使用融合模型

本次实际生成在 `model/build/bn-fusion/`，该目录由已有 `.gitignore` 忽略：

- `config.json`：原始架构配置。
- `model.safetensors`：移除 BN 参数/buffer 并加入融合 Conv bias 的完整模型权重。
- `validation_tensors.safetensors`：固定输入、原始 logits、融合 logits。

原始 timm 架构不能直接严格加载这份融合权重。用工具提供的加载函数重建
相同融合结构，再严格加载：

```python
import sys
sys.path.insert(0, "examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/model/tools")
from fuse_batchnorm import load_fused_model

model = load_fused_model(
    "examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/model/build/bn-fusion"
)
```

可用 `--checkpoint-dir`、`--output-dir`、`--validation-dir` 指定路径；
工具拒绝以原 checkpoint 目录为输出目录。

## 编译与 FPGA 状态

本次范围只有主机离线 BN folding。Triton kernel 名称、kernel case、
TTIR / Linalg MLIR / LLVM IR、ELF/BIN、FPGA build 和 `fpga_run.sh`
命令均不适用，没有生成或运行这些产物。

FPGA 实际结果：**NOT EXECUTED**。
本次 BN folding 任务状态：**COMPLETE**。
