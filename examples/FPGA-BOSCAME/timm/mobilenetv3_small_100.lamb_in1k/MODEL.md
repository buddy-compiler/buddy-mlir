# MobileNetV3 Small 100 实际 shape 清单

由 `tools/extract_shapes.py` 从指定 timm checkpoint 实跑生成。
输入 `[1, 3, 224, 224]`，NCHW、FP32、CPU、eval；输出 `[1, 1000]`。
本清单覆盖此静态 profile；不同 batch/分辨率需重新提取。预处理和输出 softmax 不属于 model.forward。

参数 2,542,856；checkpoint 张量 244；运行时 state_dict 张量 244。
模块调用 229；ATen 调用 191；去重后 110 个签名。
卷积 53（含 depthwise 11），Linear 1；卷积及 Linear MAC 总计 56,510,400。

## 主干与分类头

| 模块 | 类型 | 输入 | 输出 |
| --- | --- | --- | --- |
| <root> | MobileNetV3 | [1, 3, 224, 224] | [1, 1000] |
| conv_stem | Conv2d | [1, 3, 224, 224] | [1, 16, 112, 112] |
| bn1 | BatchNormAct2d | [1, 16, 112, 112] | [1, 16, 112, 112] |
| blocks | Sequential | [1, 16, 112, 112] | [1, 576, 7, 7] |
| blocks.0.0 | DepthwiseSeparableConv | [1, 16, 112, 112] | [1, 16, 56, 56] |
| blocks.1.0 | InvertedResidual | [1, 16, 56, 56] | [1, 24, 28, 28] |
| blocks.1.1 | InvertedResidual | [1, 24, 28, 28] | [1, 24, 28, 28] |
| blocks.2.0 | InvertedResidual | [1, 24, 28, 28] | [1, 40, 14, 14] |
| blocks.2.1 | InvertedResidual | [1, 40, 14, 14] | [1, 40, 14, 14] |
| blocks.2.2 | InvertedResidual | [1, 40, 14, 14] | [1, 40, 14, 14] |
| blocks.3.0 | InvertedResidual | [1, 40, 14, 14] | [1, 48, 14, 14] |
| blocks.3.1 | InvertedResidual | [1, 48, 14, 14] | [1, 48, 14, 14] |
| blocks.4.0 | InvertedResidual | [1, 48, 14, 14] | [1, 96, 7, 7] |
| blocks.4.1 | InvertedResidual | [1, 96, 7, 7] | [1, 96, 7, 7] |
| blocks.4.2 | InvertedResidual | [1, 96, 7, 7] | [1, 96, 7, 7] |
| blocks.5.0 | ConvBnAct | [1, 96, 7, 7] | [1, 576, 7, 7] |
| global_pool | SelectAdaptivePool2d | [1, 576, 7, 7] | [1, 576, 1, 1] |
| conv_head | Conv2d | [1, 576, 1, 1] | [1, 1024, 1, 1] |
| norm_head | Identity | [1, 1024, 1, 1] | [1, 1024, 1, 1] |
| act2 | Hardswish | [1, 1024, 1, 1] | [1, 1024, 1, 1] |
| flatten | Flatten | [1, 1024, 1, 1] | [1, 1024] |
| classifier | Linear | [1, 1024] | [1, 1000] |

## 全部卷积及全连接

权重为 OIHW；Linear 权重为 [out,in]。M,N,K 采用与 Qwen3 相同的数学约定：A[M,K] × B[K,N] → C[M,N]。
卷积为逻辑展开：M=B×Hout×Wout，N=Cout/groups，K=(Cin/groups)×Kh×Kw；每组执行一次。
depthwise 必须保留 groups，不能当作跨通道的 dense GEMM。这里只描述数学映射，尚未选择 im2col、布局变换或硬件实现。

| 模块 | 类型 | 输入 | 权重 / bias | 输出 | kernel / stride / padding | groups | M,N,K（每组） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conv_stem | conv2d | [1, 3, 224, 224] | [16, 3, 3, 3] / None | [1, 16, 112, 112] | [3, 3] / [2, 2] / [1, 1] | 1 | [12544, 16, 27] |
| blocks.0.0.conv_dw | depthwise | [1, 16, 112, 112] | [16, 1, 3, 3] / None | [1, 16, 56, 56] | [3, 3] / [2, 2] / [1, 1] | 16 | [3136, 1, 9] |
| blocks.0.0.se.conv_reduce | conv2d | [1, 16, 1, 1] | [8, 16, 1, 1] / [8] | [1, 8, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 8, 16] |
| blocks.0.0.se.conv_expand | conv2d | [1, 8, 1, 1] | [16, 8, 1, 1] / [16] | [1, 16, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 16, 8] |
| blocks.0.0.conv_pw | conv2d | [1, 16, 56, 56] | [16, 16, 1, 1] / None | [1, 16, 56, 56] | [1, 1] / [1, 1] / [0, 0] | 1 | [3136, 16, 16] |
| blocks.1.0.conv_pw | conv2d | [1, 16, 56, 56] | [72, 16, 1, 1] / None | [1, 72, 56, 56] | [1, 1] / [1, 1] / [0, 0] | 1 | [3136, 72, 16] |
| blocks.1.0.conv_dw | depthwise | [1, 72, 56, 56] | [72, 1, 3, 3] / None | [1, 72, 28, 28] | [3, 3] / [2, 2] / [1, 1] | 72 | [784, 1, 9] |
| blocks.1.0.conv_pwl | conv2d | [1, 72, 28, 28] | [24, 72, 1, 1] / None | [1, 24, 28, 28] | [1, 1] / [1, 1] / [0, 0] | 1 | [784, 24, 72] |
| blocks.1.1.conv_pw | conv2d | [1, 24, 28, 28] | [88, 24, 1, 1] / None | [1, 88, 28, 28] | [1, 1] / [1, 1] / [0, 0] | 1 | [784, 88, 24] |
| blocks.1.1.conv_dw | depthwise | [1, 88, 28, 28] | [88, 1, 3, 3] / None | [1, 88, 28, 28] | [3, 3] / [1, 1] / [1, 1] | 88 | [784, 1, 9] |
| blocks.1.1.conv_pwl | conv2d | [1, 88, 28, 28] | [24, 88, 1, 1] / None | [1, 24, 28, 28] | [1, 1] / [1, 1] / [0, 0] | 1 | [784, 24, 88] |
| blocks.2.0.conv_pw | conv2d | [1, 24, 28, 28] | [96, 24, 1, 1] / None | [1, 96, 28, 28] | [1, 1] / [1, 1] / [0, 0] | 1 | [784, 96, 24] |
| blocks.2.0.conv_dw | depthwise | [1, 96, 28, 28] | [96, 1, 5, 5] / None | [1, 96, 14, 14] | [5, 5] / [2, 2] / [2, 2] | 96 | [196, 1, 25] |
| blocks.2.0.se.conv_reduce | conv2d | [1, 96, 1, 1] | [24, 96, 1, 1] / [24] | [1, 24, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 24, 96] |
| blocks.2.0.se.conv_expand | conv2d | [1, 24, 1, 1] | [96, 24, 1, 1] / [96] | [1, 96, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 96, 24] |
| blocks.2.0.conv_pwl | conv2d | [1, 96, 14, 14] | [40, 96, 1, 1] / None | [1, 40, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 40, 96] |
| blocks.2.1.conv_pw | conv2d | [1, 40, 14, 14] | [240, 40, 1, 1] / None | [1, 240, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 240, 40] |
| blocks.2.1.conv_dw | depthwise | [1, 240, 14, 14] | [240, 1, 5, 5] / None | [1, 240, 14, 14] | [5, 5] / [1, 1] / [2, 2] | 240 | [196, 1, 25] |
| blocks.2.1.se.conv_reduce | conv2d | [1, 240, 1, 1] | [64, 240, 1, 1] / [64] | [1, 64, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 64, 240] |
| blocks.2.1.se.conv_expand | conv2d | [1, 64, 1, 1] | [240, 64, 1, 1] / [240] | [1, 240, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 240, 64] |
| blocks.2.1.conv_pwl | conv2d | [1, 240, 14, 14] | [40, 240, 1, 1] / None | [1, 40, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 40, 240] |
| blocks.2.2.conv_pw | conv2d | [1, 40, 14, 14] | [240, 40, 1, 1] / None | [1, 240, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 240, 40] |
| blocks.2.2.conv_dw | depthwise | [1, 240, 14, 14] | [240, 1, 5, 5] / None | [1, 240, 14, 14] | [5, 5] / [1, 1] / [2, 2] | 240 | [196, 1, 25] |
| blocks.2.2.se.conv_reduce | conv2d | [1, 240, 1, 1] | [64, 240, 1, 1] / [64] | [1, 64, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 64, 240] |
| blocks.2.2.se.conv_expand | conv2d | [1, 64, 1, 1] | [240, 64, 1, 1] / [240] | [1, 240, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 240, 64] |
| blocks.2.2.conv_pwl | conv2d | [1, 240, 14, 14] | [40, 240, 1, 1] / None | [1, 40, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 40, 240] |
| blocks.3.0.conv_pw | conv2d | [1, 40, 14, 14] | [120, 40, 1, 1] / None | [1, 120, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 120, 40] |
| blocks.3.0.conv_dw | depthwise | [1, 120, 14, 14] | [120, 1, 5, 5] / None | [1, 120, 14, 14] | [5, 5] / [1, 1] / [2, 2] | 120 | [196, 1, 25] |
| blocks.3.0.se.conv_reduce | conv2d | [1, 120, 1, 1] | [32, 120, 1, 1] / [32] | [1, 32, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 32, 120] |
| blocks.3.0.se.conv_expand | conv2d | [1, 32, 1, 1] | [120, 32, 1, 1] / [120] | [1, 120, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 120, 32] |
| blocks.3.0.conv_pwl | conv2d | [1, 120, 14, 14] | [48, 120, 1, 1] / None | [1, 48, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 48, 120] |
| blocks.3.1.conv_pw | conv2d | [1, 48, 14, 14] | [144, 48, 1, 1] / None | [1, 144, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 144, 48] |
| blocks.3.1.conv_dw | depthwise | [1, 144, 14, 14] | [144, 1, 5, 5] / None | [1, 144, 14, 14] | [5, 5] / [1, 1] / [2, 2] | 144 | [196, 1, 25] |
| blocks.3.1.se.conv_reduce | conv2d | [1, 144, 1, 1] | [40, 144, 1, 1] / [40] | [1, 40, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 40, 144] |
| blocks.3.1.se.conv_expand | conv2d | [1, 40, 1, 1] | [144, 40, 1, 1] / [144] | [1, 144, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 144, 40] |
| blocks.3.1.conv_pwl | conv2d | [1, 144, 14, 14] | [48, 144, 1, 1] / None | [1, 48, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 48, 144] |
| blocks.4.0.conv_pw | conv2d | [1, 48, 14, 14] | [288, 48, 1, 1] / None | [1, 288, 14, 14] | [1, 1] / [1, 1] / [0, 0] | 1 | [196, 288, 48] |
| blocks.4.0.conv_dw | depthwise | [1, 288, 14, 14] | [288, 1, 5, 5] / None | [1, 288, 7, 7] | [5, 5] / [2, 2] / [2, 2] | 288 | [49, 1, 25] |
| blocks.4.0.se.conv_reduce | conv2d | [1, 288, 1, 1] | [72, 288, 1, 1] / [72] | [1, 72, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 72, 288] |
| blocks.4.0.se.conv_expand | conv2d | [1, 72, 1, 1] | [288, 72, 1, 1] / [288] | [1, 288, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 288, 72] |
| blocks.4.0.conv_pwl | conv2d | [1, 288, 7, 7] | [96, 288, 1, 1] / None | [1, 96, 7, 7] | [1, 1] / [1, 1] / [0, 0] | 1 | [49, 96, 288] |
| blocks.4.1.conv_pw | conv2d | [1, 96, 7, 7] | [576, 96, 1, 1] / None | [1, 576, 7, 7] | [1, 1] / [1, 1] / [0, 0] | 1 | [49, 576, 96] |
| blocks.4.1.conv_dw | depthwise | [1, 576, 7, 7] | [576, 1, 5, 5] / None | [1, 576, 7, 7] | [5, 5] / [1, 1] / [2, 2] | 576 | [49, 1, 25] |
| blocks.4.1.se.conv_reduce | conv2d | [1, 576, 1, 1] | [144, 576, 1, 1] / [144] | [1, 144, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 144, 576] |
| blocks.4.1.se.conv_expand | conv2d | [1, 144, 1, 1] | [576, 144, 1, 1] / [576] | [1, 576, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 576, 144] |
| blocks.4.1.conv_pwl | conv2d | [1, 576, 7, 7] | [96, 576, 1, 1] / None | [1, 96, 7, 7] | [1, 1] / [1, 1] / [0, 0] | 1 | [49, 96, 576] |
| blocks.4.2.conv_pw | conv2d | [1, 96, 7, 7] | [576, 96, 1, 1] / None | [1, 576, 7, 7] | [1, 1] / [1, 1] / [0, 0] | 1 | [49, 576, 96] |
| blocks.4.2.conv_dw | depthwise | [1, 576, 7, 7] | [576, 1, 5, 5] / None | [1, 576, 7, 7] | [5, 5] / [1, 1] / [2, 2] | 576 | [49, 1, 25] |
| blocks.4.2.se.conv_reduce | conv2d | [1, 576, 1, 1] | [144, 576, 1, 1] / [144] | [1, 144, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 144, 576] |
| blocks.4.2.se.conv_expand | conv2d | [1, 144, 1, 1] | [576, 144, 1, 1] / [576] | [1, 576, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 576, 144] |
| blocks.4.2.conv_pwl | conv2d | [1, 576, 7, 7] | [96, 576, 1, 1] / None | [1, 96, 7, 7] | [1, 1] / [1, 1] / [0, 0] | 1 | [49, 96, 576] |
| blocks.5.0.conv | conv2d | [1, 96, 7, 7] | [576, 96, 1, 1] / None | [1, 576, 7, 7] | [1, 1] / [1, 1] / [0, 0] | 1 | [49, 576, 96] |
| conv_head | conv2d | [1, 576, 1, 1] | [1024, 576, 1, 1] / [1024] | [1, 1024, 1, 1] | [1, 1] / [1, 1] / [0, 0] | 1 | [1, 1024, 576] |
| classifier | linear | [1, 1024] | [1000, 1024] / [1000] | [1, 1000] | — / — / — | 1 | [1, 1000, 1024] |

## ATen 算子统计

包括 SE 的 mean / 广播乘法、残差 add、激活、BN、池化、flatten，以及实际出现的辅助操作。
BN 的空输出张量、empty、视图等按运行时原样记录，不代表均需独立 FPGA 内核。

| ATen | 调用次数 |
| --- | ---: |
| aten.add.Tensor | 6 |
| aten.addmm.default | 1 |
| aten.convolution.default | 53 |
| aten.empty.memory_format | 34 |
| aten.hardsigmoid.default | 9 |
| aten.hardswish_.default | 19 |
| aten.mean.dim | 10 |
| aten.mul.Tensor | 9 |
| aten.native_batch_norm.default | 34 |
| aten.relu_.default | 14 |
| aten.t.default | 1 |
| aten.view.default | 1 |

## 完整 ATen 执行序列

表中只展示张量 shape；标量参数、归约轴、keepdim、广播输入、dtype、stride 和 BN epsilon 详见 model.json。

| 序号 | 所属模块 | ATen | 输入张量 | 输出张量 |
| ---: | --- | --- | --- | --- |
| 0 | conv_stem | aten.convolution.default | [1, 3, 224, 224], [16, 3, 3, 3] | [1, 16, 112, 112] |
| 1 | bn1 | aten.empty.memory_format | — | [0] |
| 2 | bn1 | aten.native_batch_norm.default | [1, 16, 112, 112], [16], [16], [16], [16] | [1, 16, 112, 112], [0], [0] |
| 3 | bn1.act | aten.hardswish_.default | [1, 16, 112, 112] | [1, 16, 112, 112] |
| 4 | blocks.0.0.conv_dw | aten.convolution.default | [1, 16, 112, 112], [16, 1, 3, 3] | [1, 16, 56, 56] |
| 5 | blocks.0.0.bn1 | aten.empty.memory_format | — | [0] |
| 6 | blocks.0.0.bn1 | aten.native_batch_norm.default | [1, 16, 56, 56], [16], [16], [16], [16] | [1, 16, 56, 56], [0], [0] |
| 7 | blocks.0.0.bn1.act | aten.relu_.default | [1, 16, 56, 56] | [1, 16, 56, 56] |
| 8 | blocks.0.0.se | aten.mean.dim | [1, 16, 56, 56] | [1, 16, 1, 1] |
| 9 | blocks.0.0.se.conv_reduce | aten.convolution.default | [1, 16, 1, 1], [8, 16, 1, 1], [8] | [1, 8, 1, 1] |
| 10 | blocks.0.0.se.act1 | aten.relu_.default | [1, 8, 1, 1] | [1, 8, 1, 1] |
| 11 | blocks.0.0.se.conv_expand | aten.convolution.default | [1, 8, 1, 1], [16, 8, 1, 1], [16] | [1, 16, 1, 1] |
| 12 | blocks.0.0.se.gate | aten.hardsigmoid.default | [1, 16, 1, 1] | [1, 16, 1, 1] |
| 13 | blocks.0.0.se | aten.mul.Tensor | [1, 16, 56, 56], [1, 16, 1, 1] | [1, 16, 56, 56] |
| 14 | blocks.0.0.conv_pw | aten.convolution.default | [1, 16, 56, 56], [16, 16, 1, 1] | [1, 16, 56, 56] |
| 15 | blocks.0.0.bn2 | aten.empty.memory_format | — | [0] |
| 16 | blocks.0.0.bn2 | aten.native_batch_norm.default | [1, 16, 56, 56], [16], [16], [16], [16] | [1, 16, 56, 56], [0], [0] |
| 17 | blocks.1.0.conv_pw | aten.convolution.default | [1, 16, 56, 56], [72, 16, 1, 1] | [1, 72, 56, 56] |
| 18 | blocks.1.0.bn1 | aten.empty.memory_format | — | [0] |
| 19 | blocks.1.0.bn1 | aten.native_batch_norm.default | [1, 72, 56, 56], [72], [72], [72], [72] | [1, 72, 56, 56], [0], [0] |
| 20 | blocks.1.0.bn1.act | aten.relu_.default | [1, 72, 56, 56] | [1, 72, 56, 56] |
| 21 | blocks.1.0.conv_dw | aten.convolution.default | [1, 72, 56, 56], [72, 1, 3, 3] | [1, 72, 28, 28] |
| 22 | blocks.1.0.bn2 | aten.empty.memory_format | — | [0] |
| 23 | blocks.1.0.bn2 | aten.native_batch_norm.default | [1, 72, 28, 28], [72], [72], [72], [72] | [1, 72, 28, 28], [0], [0] |
| 24 | blocks.1.0.bn2.act | aten.relu_.default | [1, 72, 28, 28] | [1, 72, 28, 28] |
| 25 | blocks.1.0.conv_pwl | aten.convolution.default | [1, 72, 28, 28], [24, 72, 1, 1] | [1, 24, 28, 28] |
| 26 | blocks.1.0.bn3 | aten.empty.memory_format | — | [0] |
| 27 | blocks.1.0.bn3 | aten.native_batch_norm.default | [1, 24, 28, 28], [24], [24], [24], [24] | [1, 24, 28, 28], [0], [0] |
| 28 | blocks.1.1.conv_pw | aten.convolution.default | [1, 24, 28, 28], [88, 24, 1, 1] | [1, 88, 28, 28] |
| 29 | blocks.1.1.bn1 | aten.empty.memory_format | — | [0] |
| 30 | blocks.1.1.bn1 | aten.native_batch_norm.default | [1, 88, 28, 28], [88], [88], [88], [88] | [1, 88, 28, 28], [0], [0] |
| 31 | blocks.1.1.bn1.act | aten.relu_.default | [1, 88, 28, 28] | [1, 88, 28, 28] |
| 32 | blocks.1.1.conv_dw | aten.convolution.default | [1, 88, 28, 28], [88, 1, 3, 3] | [1, 88, 28, 28] |
| 33 | blocks.1.1.bn2 | aten.empty.memory_format | — | [0] |
| 34 | blocks.1.1.bn2 | aten.native_batch_norm.default | [1, 88, 28, 28], [88], [88], [88], [88] | [1, 88, 28, 28], [0], [0] |
| 35 | blocks.1.1.bn2.act | aten.relu_.default | [1, 88, 28, 28] | [1, 88, 28, 28] |
| 36 | blocks.1.1.conv_pwl | aten.convolution.default | [1, 88, 28, 28], [24, 88, 1, 1] | [1, 24, 28, 28] |
| 37 | blocks.1.1.bn3 | aten.empty.memory_format | — | [0] |
| 38 | blocks.1.1.bn3 | aten.native_batch_norm.default | [1, 24, 28, 28], [24], [24], [24], [24] | [1, 24, 28, 28], [0], [0] |
| 39 | blocks.1.1 | aten.add.Tensor | [1, 24, 28, 28], [1, 24, 28, 28] | [1, 24, 28, 28] |
| 40 | blocks.2.0.conv_pw | aten.convolution.default | [1, 24, 28, 28], [96, 24, 1, 1] | [1, 96, 28, 28] |
| 41 | blocks.2.0.bn1 | aten.empty.memory_format | — | [0] |
| 42 | blocks.2.0.bn1 | aten.native_batch_norm.default | [1, 96, 28, 28], [96], [96], [96], [96] | [1, 96, 28, 28], [0], [0] |
| 43 | blocks.2.0.bn1.act | aten.hardswish_.default | [1, 96, 28, 28] | [1, 96, 28, 28] |
| 44 | blocks.2.0.conv_dw | aten.convolution.default | [1, 96, 28, 28], [96, 1, 5, 5] | [1, 96, 14, 14] |
| 45 | blocks.2.0.bn2 | aten.empty.memory_format | — | [0] |
| 46 | blocks.2.0.bn2 | aten.native_batch_norm.default | [1, 96, 14, 14], [96], [96], [96], [96] | [1, 96, 14, 14], [0], [0] |
| 47 | blocks.2.0.bn2.act | aten.hardswish_.default | [1, 96, 14, 14] | [1, 96, 14, 14] |
| 48 | blocks.2.0.se | aten.mean.dim | [1, 96, 14, 14] | [1, 96, 1, 1] |
| 49 | blocks.2.0.se.conv_reduce | aten.convolution.default | [1, 96, 1, 1], [24, 96, 1, 1], [24] | [1, 24, 1, 1] |
| 50 | blocks.2.0.se.act1 | aten.relu_.default | [1, 24, 1, 1] | [1, 24, 1, 1] |
| 51 | blocks.2.0.se.conv_expand | aten.convolution.default | [1, 24, 1, 1], [96, 24, 1, 1], [96] | [1, 96, 1, 1] |
| 52 | blocks.2.0.se.gate | aten.hardsigmoid.default | [1, 96, 1, 1] | [1, 96, 1, 1] |
| 53 | blocks.2.0.se | aten.mul.Tensor | [1, 96, 14, 14], [1, 96, 1, 1] | [1, 96, 14, 14] |
| 54 | blocks.2.0.conv_pwl | aten.convolution.default | [1, 96, 14, 14], [40, 96, 1, 1] | [1, 40, 14, 14] |
| 55 | blocks.2.0.bn3 | aten.empty.memory_format | — | [0] |
| 56 | blocks.2.0.bn3 | aten.native_batch_norm.default | [1, 40, 14, 14], [40], [40], [40], [40] | [1, 40, 14, 14], [0], [0] |
| 57 | blocks.2.1.conv_pw | aten.convolution.default | [1, 40, 14, 14], [240, 40, 1, 1] | [1, 240, 14, 14] |
| 58 | blocks.2.1.bn1 | aten.empty.memory_format | — | [0] |
| 59 | blocks.2.1.bn1 | aten.native_batch_norm.default | [1, 240, 14, 14], [240], [240], [240], [240] | [1, 240, 14, 14], [0], [0] |
| 60 | blocks.2.1.bn1.act | aten.hardswish_.default | [1, 240, 14, 14] | [1, 240, 14, 14] |
| 61 | blocks.2.1.conv_dw | aten.convolution.default | [1, 240, 14, 14], [240, 1, 5, 5] | [1, 240, 14, 14] |
| 62 | blocks.2.1.bn2 | aten.empty.memory_format | — | [0] |
| 63 | blocks.2.1.bn2 | aten.native_batch_norm.default | [1, 240, 14, 14], [240], [240], [240], [240] | [1, 240, 14, 14], [0], [0] |
| 64 | blocks.2.1.bn2.act | aten.hardswish_.default | [1, 240, 14, 14] | [1, 240, 14, 14] |
| 65 | blocks.2.1.se | aten.mean.dim | [1, 240, 14, 14] | [1, 240, 1, 1] |
| 66 | blocks.2.1.se.conv_reduce | aten.convolution.default | [1, 240, 1, 1], [64, 240, 1, 1], [64] | [1, 64, 1, 1] |
| 67 | blocks.2.1.se.act1 | aten.relu_.default | [1, 64, 1, 1] | [1, 64, 1, 1] |
| 68 | blocks.2.1.se.conv_expand | aten.convolution.default | [1, 64, 1, 1], [240, 64, 1, 1], [240] | [1, 240, 1, 1] |
| 69 | blocks.2.1.se.gate | aten.hardsigmoid.default | [1, 240, 1, 1] | [1, 240, 1, 1] |
| 70 | blocks.2.1.se | aten.mul.Tensor | [1, 240, 14, 14], [1, 240, 1, 1] | [1, 240, 14, 14] |
| 71 | blocks.2.1.conv_pwl | aten.convolution.default | [1, 240, 14, 14], [40, 240, 1, 1] | [1, 40, 14, 14] |
| 72 | blocks.2.1.bn3 | aten.empty.memory_format | — | [0] |
| 73 | blocks.2.1.bn3 | aten.native_batch_norm.default | [1, 40, 14, 14], [40], [40], [40], [40] | [1, 40, 14, 14], [0], [0] |
| 74 | blocks.2.1 | aten.add.Tensor | [1, 40, 14, 14], [1, 40, 14, 14] | [1, 40, 14, 14] |
| 75 | blocks.2.2.conv_pw | aten.convolution.default | [1, 40, 14, 14], [240, 40, 1, 1] | [1, 240, 14, 14] |
| 76 | blocks.2.2.bn1 | aten.empty.memory_format | — | [0] |
| 77 | blocks.2.2.bn1 | aten.native_batch_norm.default | [1, 240, 14, 14], [240], [240], [240], [240] | [1, 240, 14, 14], [0], [0] |
| 78 | blocks.2.2.bn1.act | aten.hardswish_.default | [1, 240, 14, 14] | [1, 240, 14, 14] |
| 79 | blocks.2.2.conv_dw | aten.convolution.default | [1, 240, 14, 14], [240, 1, 5, 5] | [1, 240, 14, 14] |
| 80 | blocks.2.2.bn2 | aten.empty.memory_format | — | [0] |
| 81 | blocks.2.2.bn2 | aten.native_batch_norm.default | [1, 240, 14, 14], [240], [240], [240], [240] | [1, 240, 14, 14], [0], [0] |
| 82 | blocks.2.2.bn2.act | aten.hardswish_.default | [1, 240, 14, 14] | [1, 240, 14, 14] |
| 83 | blocks.2.2.se | aten.mean.dim | [1, 240, 14, 14] | [1, 240, 1, 1] |
| 84 | blocks.2.2.se.conv_reduce | aten.convolution.default | [1, 240, 1, 1], [64, 240, 1, 1], [64] | [1, 64, 1, 1] |
| 85 | blocks.2.2.se.act1 | aten.relu_.default | [1, 64, 1, 1] | [1, 64, 1, 1] |
| 86 | blocks.2.2.se.conv_expand | aten.convolution.default | [1, 64, 1, 1], [240, 64, 1, 1], [240] | [1, 240, 1, 1] |
| 87 | blocks.2.2.se.gate | aten.hardsigmoid.default | [1, 240, 1, 1] | [1, 240, 1, 1] |
| 88 | blocks.2.2.se | aten.mul.Tensor | [1, 240, 14, 14], [1, 240, 1, 1] | [1, 240, 14, 14] |
| 89 | blocks.2.2.conv_pwl | aten.convolution.default | [1, 240, 14, 14], [40, 240, 1, 1] | [1, 40, 14, 14] |
| 90 | blocks.2.2.bn3 | aten.empty.memory_format | — | [0] |
| 91 | blocks.2.2.bn3 | aten.native_batch_norm.default | [1, 40, 14, 14], [40], [40], [40], [40] | [1, 40, 14, 14], [0], [0] |
| 92 | blocks.2.2 | aten.add.Tensor | [1, 40, 14, 14], [1, 40, 14, 14] | [1, 40, 14, 14] |
| 93 | blocks.3.0.conv_pw | aten.convolution.default | [1, 40, 14, 14], [120, 40, 1, 1] | [1, 120, 14, 14] |
| 94 | blocks.3.0.bn1 | aten.empty.memory_format | — | [0] |
| 95 | blocks.3.0.bn1 | aten.native_batch_norm.default | [1, 120, 14, 14], [120], [120], [120], [120] | [1, 120, 14, 14], [0], [0] |
| 96 | blocks.3.0.bn1.act | aten.hardswish_.default | [1, 120, 14, 14] | [1, 120, 14, 14] |
| 97 | blocks.3.0.conv_dw | aten.convolution.default | [1, 120, 14, 14], [120, 1, 5, 5] | [1, 120, 14, 14] |
| 98 | blocks.3.0.bn2 | aten.empty.memory_format | — | [0] |
| 99 | blocks.3.0.bn2 | aten.native_batch_norm.default | [1, 120, 14, 14], [120], [120], [120], [120] | [1, 120, 14, 14], [0], [0] |
| 100 | blocks.3.0.bn2.act | aten.hardswish_.default | [1, 120, 14, 14] | [1, 120, 14, 14] |
| 101 | blocks.3.0.se | aten.mean.dim | [1, 120, 14, 14] | [1, 120, 1, 1] |
| 102 | blocks.3.0.se.conv_reduce | aten.convolution.default | [1, 120, 1, 1], [32, 120, 1, 1], [32] | [1, 32, 1, 1] |
| 103 | blocks.3.0.se.act1 | aten.relu_.default | [1, 32, 1, 1] | [1, 32, 1, 1] |
| 104 | blocks.3.0.se.conv_expand | aten.convolution.default | [1, 32, 1, 1], [120, 32, 1, 1], [120] | [1, 120, 1, 1] |
| 105 | blocks.3.0.se.gate | aten.hardsigmoid.default | [1, 120, 1, 1] | [1, 120, 1, 1] |
| 106 | blocks.3.0.se | aten.mul.Tensor | [1, 120, 14, 14], [1, 120, 1, 1] | [1, 120, 14, 14] |
| 107 | blocks.3.0.conv_pwl | aten.convolution.default | [1, 120, 14, 14], [48, 120, 1, 1] | [1, 48, 14, 14] |
| 108 | blocks.3.0.bn3 | aten.empty.memory_format | — | [0] |
| 109 | blocks.3.0.bn3 | aten.native_batch_norm.default | [1, 48, 14, 14], [48], [48], [48], [48] | [1, 48, 14, 14], [0], [0] |
| 110 | blocks.3.1.conv_pw | aten.convolution.default | [1, 48, 14, 14], [144, 48, 1, 1] | [1, 144, 14, 14] |
| 111 | blocks.3.1.bn1 | aten.empty.memory_format | — | [0] |
| 112 | blocks.3.1.bn1 | aten.native_batch_norm.default | [1, 144, 14, 14], [144], [144], [144], [144] | [1, 144, 14, 14], [0], [0] |
| 113 | blocks.3.1.bn1.act | aten.hardswish_.default | [1, 144, 14, 14] | [1, 144, 14, 14] |
| 114 | blocks.3.1.conv_dw | aten.convolution.default | [1, 144, 14, 14], [144, 1, 5, 5] | [1, 144, 14, 14] |
| 115 | blocks.3.1.bn2 | aten.empty.memory_format | — | [0] |
| 116 | blocks.3.1.bn2 | aten.native_batch_norm.default | [1, 144, 14, 14], [144], [144], [144], [144] | [1, 144, 14, 14], [0], [0] |
| 117 | blocks.3.1.bn2.act | aten.hardswish_.default | [1, 144, 14, 14] | [1, 144, 14, 14] |
| 118 | blocks.3.1.se | aten.mean.dim | [1, 144, 14, 14] | [1, 144, 1, 1] |
| 119 | blocks.3.1.se.conv_reduce | aten.convolution.default | [1, 144, 1, 1], [40, 144, 1, 1], [40] | [1, 40, 1, 1] |
| 120 | blocks.3.1.se.act1 | aten.relu_.default | [1, 40, 1, 1] | [1, 40, 1, 1] |
| 121 | blocks.3.1.se.conv_expand | aten.convolution.default | [1, 40, 1, 1], [144, 40, 1, 1], [144] | [1, 144, 1, 1] |
| 122 | blocks.3.1.se.gate | aten.hardsigmoid.default | [1, 144, 1, 1] | [1, 144, 1, 1] |
| 123 | blocks.3.1.se | aten.mul.Tensor | [1, 144, 14, 14], [1, 144, 1, 1] | [1, 144, 14, 14] |
| 124 | blocks.3.1.conv_pwl | aten.convolution.default | [1, 144, 14, 14], [48, 144, 1, 1] | [1, 48, 14, 14] |
| 125 | blocks.3.1.bn3 | aten.empty.memory_format | — | [0] |
| 126 | blocks.3.1.bn3 | aten.native_batch_norm.default | [1, 48, 14, 14], [48], [48], [48], [48] | [1, 48, 14, 14], [0], [0] |
| 127 | blocks.3.1 | aten.add.Tensor | [1, 48, 14, 14], [1, 48, 14, 14] | [1, 48, 14, 14] |
| 128 | blocks.4.0.conv_pw | aten.convolution.default | [1, 48, 14, 14], [288, 48, 1, 1] | [1, 288, 14, 14] |
| 129 | blocks.4.0.bn1 | aten.empty.memory_format | — | [0] |
| 130 | blocks.4.0.bn1 | aten.native_batch_norm.default | [1, 288, 14, 14], [288], [288], [288], [288] | [1, 288, 14, 14], [0], [0] |
| 131 | blocks.4.0.bn1.act | aten.hardswish_.default | [1, 288, 14, 14] | [1, 288, 14, 14] |
| 132 | blocks.4.0.conv_dw | aten.convolution.default | [1, 288, 14, 14], [288, 1, 5, 5] | [1, 288, 7, 7] |
| 133 | blocks.4.0.bn2 | aten.empty.memory_format | — | [0] |
| 134 | blocks.4.0.bn2 | aten.native_batch_norm.default | [1, 288, 7, 7], [288], [288], [288], [288] | [1, 288, 7, 7], [0], [0] |
| 135 | blocks.4.0.bn2.act | aten.hardswish_.default | [1, 288, 7, 7] | [1, 288, 7, 7] |
| 136 | blocks.4.0.se | aten.mean.dim | [1, 288, 7, 7] | [1, 288, 1, 1] |
| 137 | blocks.4.0.se.conv_reduce | aten.convolution.default | [1, 288, 1, 1], [72, 288, 1, 1], [72] | [1, 72, 1, 1] |
| 138 | blocks.4.0.se.act1 | aten.relu_.default | [1, 72, 1, 1] | [1, 72, 1, 1] |
| 139 | blocks.4.0.se.conv_expand | aten.convolution.default | [1, 72, 1, 1], [288, 72, 1, 1], [288] | [1, 288, 1, 1] |
| 140 | blocks.4.0.se.gate | aten.hardsigmoid.default | [1, 288, 1, 1] | [1, 288, 1, 1] |
| 141 | blocks.4.0.se | aten.mul.Tensor | [1, 288, 7, 7], [1, 288, 1, 1] | [1, 288, 7, 7] |
| 142 | blocks.4.0.conv_pwl | aten.convolution.default | [1, 288, 7, 7], [96, 288, 1, 1] | [1, 96, 7, 7] |
| 143 | blocks.4.0.bn3 | aten.empty.memory_format | — | [0] |
| 144 | blocks.4.0.bn3 | aten.native_batch_norm.default | [1, 96, 7, 7], [96], [96], [96], [96] | [1, 96, 7, 7], [0], [0] |
| 145 | blocks.4.1.conv_pw | aten.convolution.default | [1, 96, 7, 7], [576, 96, 1, 1] | [1, 576, 7, 7] |
| 146 | blocks.4.1.bn1 | aten.empty.memory_format | — | [0] |
| 147 | blocks.4.1.bn1 | aten.native_batch_norm.default | [1, 576, 7, 7], [576], [576], [576], [576] | [1, 576, 7, 7], [0], [0] |
| 148 | blocks.4.1.bn1.act | aten.hardswish_.default | [1, 576, 7, 7] | [1, 576, 7, 7] |
| 149 | blocks.4.1.conv_dw | aten.convolution.default | [1, 576, 7, 7], [576, 1, 5, 5] | [1, 576, 7, 7] |
| 150 | blocks.4.1.bn2 | aten.empty.memory_format | — | [0] |
| 151 | blocks.4.1.bn2 | aten.native_batch_norm.default | [1, 576, 7, 7], [576], [576], [576], [576] | [1, 576, 7, 7], [0], [0] |
| 152 | blocks.4.1.bn2.act | aten.hardswish_.default | [1, 576, 7, 7] | [1, 576, 7, 7] |
| 153 | blocks.4.1.se | aten.mean.dim | [1, 576, 7, 7] | [1, 576, 1, 1] |
| 154 | blocks.4.1.se.conv_reduce | aten.convolution.default | [1, 576, 1, 1], [144, 576, 1, 1], [144] | [1, 144, 1, 1] |
| 155 | blocks.4.1.se.act1 | aten.relu_.default | [1, 144, 1, 1] | [1, 144, 1, 1] |
| 156 | blocks.4.1.se.conv_expand | aten.convolution.default | [1, 144, 1, 1], [576, 144, 1, 1], [576] | [1, 576, 1, 1] |
| 157 | blocks.4.1.se.gate | aten.hardsigmoid.default | [1, 576, 1, 1] | [1, 576, 1, 1] |
| 158 | blocks.4.1.se | aten.mul.Tensor | [1, 576, 7, 7], [1, 576, 1, 1] | [1, 576, 7, 7] |
| 159 | blocks.4.1.conv_pwl | aten.convolution.default | [1, 576, 7, 7], [96, 576, 1, 1] | [1, 96, 7, 7] |
| 160 | blocks.4.1.bn3 | aten.empty.memory_format | — | [0] |
| 161 | blocks.4.1.bn3 | aten.native_batch_norm.default | [1, 96, 7, 7], [96], [96], [96], [96] | [1, 96, 7, 7], [0], [0] |
| 162 | blocks.4.1 | aten.add.Tensor | [1, 96, 7, 7], [1, 96, 7, 7] | [1, 96, 7, 7] |
| 163 | blocks.4.2.conv_pw | aten.convolution.default | [1, 96, 7, 7], [576, 96, 1, 1] | [1, 576, 7, 7] |
| 164 | blocks.4.2.bn1 | aten.empty.memory_format | — | [0] |
| 165 | blocks.4.2.bn1 | aten.native_batch_norm.default | [1, 576, 7, 7], [576], [576], [576], [576] | [1, 576, 7, 7], [0], [0] |
| 166 | blocks.4.2.bn1.act | aten.hardswish_.default | [1, 576, 7, 7] | [1, 576, 7, 7] |
| 167 | blocks.4.2.conv_dw | aten.convolution.default | [1, 576, 7, 7], [576, 1, 5, 5] | [1, 576, 7, 7] |
| 168 | blocks.4.2.bn2 | aten.empty.memory_format | — | [0] |
| 169 | blocks.4.2.bn2 | aten.native_batch_norm.default | [1, 576, 7, 7], [576], [576], [576], [576] | [1, 576, 7, 7], [0], [0] |
| 170 | blocks.4.2.bn2.act | aten.hardswish_.default | [1, 576, 7, 7] | [1, 576, 7, 7] |
| 171 | blocks.4.2.se | aten.mean.dim | [1, 576, 7, 7] | [1, 576, 1, 1] |
| 172 | blocks.4.2.se.conv_reduce | aten.convolution.default | [1, 576, 1, 1], [144, 576, 1, 1], [144] | [1, 144, 1, 1] |
| 173 | blocks.4.2.se.act1 | aten.relu_.default | [1, 144, 1, 1] | [1, 144, 1, 1] |
| 174 | blocks.4.2.se.conv_expand | aten.convolution.default | [1, 144, 1, 1], [576, 144, 1, 1], [576] | [1, 576, 1, 1] |
| 175 | blocks.4.2.se.gate | aten.hardsigmoid.default | [1, 576, 1, 1] | [1, 576, 1, 1] |
| 176 | blocks.4.2.se | aten.mul.Tensor | [1, 576, 7, 7], [1, 576, 1, 1] | [1, 576, 7, 7] |
| 177 | blocks.4.2.conv_pwl | aten.convolution.default | [1, 576, 7, 7], [96, 576, 1, 1] | [1, 96, 7, 7] |
| 178 | blocks.4.2.bn3 | aten.empty.memory_format | — | [0] |
| 179 | blocks.4.2.bn3 | aten.native_batch_norm.default | [1, 96, 7, 7], [96], [96], [96], [96] | [1, 96, 7, 7], [0], [0] |
| 180 | blocks.4.2 | aten.add.Tensor | [1, 96, 7, 7], [1, 96, 7, 7] | [1, 96, 7, 7] |
| 181 | blocks.5.0.conv | aten.convolution.default | [1, 96, 7, 7], [576, 96, 1, 1] | [1, 576, 7, 7] |
| 182 | blocks.5.0.bn1 | aten.empty.memory_format | — | [0] |
| 183 | blocks.5.0.bn1 | aten.native_batch_norm.default | [1, 576, 7, 7], [576], [576], [576], [576] | [1, 576, 7, 7], [0], [0] |
| 184 | blocks.5.0.bn1.act | aten.hardswish_.default | [1, 576, 7, 7] | [1, 576, 7, 7] |
| 185 | global_pool.pool | aten.mean.dim | [1, 576, 7, 7] | [1, 576, 1, 1] |
| 186 | conv_head | aten.convolution.default | [1, 576, 1, 1], [1024, 576, 1, 1], [1024] | [1, 1024, 1, 1] |
| 187 | act2 | aten.hardswish_.default | [1, 1024, 1, 1] | [1, 1024, 1, 1] |
| 188 | flatten | aten.view.default | [1, 1024, 1, 1] | [1, 1024] |
| 189 | classifier | aten.t.default | [1000, 1024] | [1024, 1000] |
| 190 | classifier | aten.addmm.default | [1000], [1, 1024], [1024, 1000] | [1, 1000] |

## 全部参数及 buffer

BN num_batches_tracked 是整数标量 buffer，不参与 eval 计算；表中区分参数、buffer 及 checkpoint 来源。

| 名称 | 类型 | shape | dtype | checkpoint 中存在 |
| --- | --- | --- | --- | --- |
| conv_stem.weight | parameter | [16, 3, 3, 3] | float32 | True |
| bn1.weight | parameter | [16] | float32 | True |
| bn1.bias | parameter | [16] | float32 | True |
| bn1.running_mean | buffer | [16] | float32 | True |
| bn1.running_var | buffer | [16] | float32 | True |
| bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.0.0.conv_dw.weight | parameter | [16, 1, 3, 3] | float32 | True |
| blocks.0.0.bn1.weight | parameter | [16] | float32 | True |
| blocks.0.0.bn1.bias | parameter | [16] | float32 | True |
| blocks.0.0.bn1.running_mean | buffer | [16] | float32 | True |
| blocks.0.0.bn1.running_var | buffer | [16] | float32 | True |
| blocks.0.0.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.0.0.se.conv_reduce.weight | parameter | [8, 16, 1, 1] | float32 | True |
| blocks.0.0.se.conv_reduce.bias | parameter | [8] | float32 | True |
| blocks.0.0.se.conv_expand.weight | parameter | [16, 8, 1, 1] | float32 | True |
| blocks.0.0.se.conv_expand.bias | parameter | [16] | float32 | True |
| blocks.0.0.conv_pw.weight | parameter | [16, 16, 1, 1] | float32 | True |
| blocks.0.0.bn2.weight | parameter | [16] | float32 | True |
| blocks.0.0.bn2.bias | parameter | [16] | float32 | True |
| blocks.0.0.bn2.running_mean | buffer | [16] | float32 | True |
| blocks.0.0.bn2.running_var | buffer | [16] | float32 | True |
| blocks.0.0.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.1.0.conv_pw.weight | parameter | [72, 16, 1, 1] | float32 | True |
| blocks.1.0.bn1.weight | parameter | [72] | float32 | True |
| blocks.1.0.bn1.bias | parameter | [72] | float32 | True |
| blocks.1.0.bn1.running_mean | buffer | [72] | float32 | True |
| blocks.1.0.bn1.running_var | buffer | [72] | float32 | True |
| blocks.1.0.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.1.0.conv_dw.weight | parameter | [72, 1, 3, 3] | float32 | True |
| blocks.1.0.bn2.weight | parameter | [72] | float32 | True |
| blocks.1.0.bn2.bias | parameter | [72] | float32 | True |
| blocks.1.0.bn2.running_mean | buffer | [72] | float32 | True |
| blocks.1.0.bn2.running_var | buffer | [72] | float32 | True |
| blocks.1.0.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.1.0.conv_pwl.weight | parameter | [24, 72, 1, 1] | float32 | True |
| blocks.1.0.bn3.weight | parameter | [24] | float32 | True |
| blocks.1.0.bn3.bias | parameter | [24] | float32 | True |
| blocks.1.0.bn3.running_mean | buffer | [24] | float32 | True |
| blocks.1.0.bn3.running_var | buffer | [24] | float32 | True |
| blocks.1.0.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.1.1.conv_pw.weight | parameter | [88, 24, 1, 1] | float32 | True |
| blocks.1.1.bn1.weight | parameter | [88] | float32 | True |
| blocks.1.1.bn1.bias | parameter | [88] | float32 | True |
| blocks.1.1.bn1.running_mean | buffer | [88] | float32 | True |
| blocks.1.1.bn1.running_var | buffer | [88] | float32 | True |
| blocks.1.1.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.1.1.conv_dw.weight | parameter | [88, 1, 3, 3] | float32 | True |
| blocks.1.1.bn2.weight | parameter | [88] | float32 | True |
| blocks.1.1.bn2.bias | parameter | [88] | float32 | True |
| blocks.1.1.bn2.running_mean | buffer | [88] | float32 | True |
| blocks.1.1.bn2.running_var | buffer | [88] | float32 | True |
| blocks.1.1.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.1.1.conv_pwl.weight | parameter | [24, 88, 1, 1] | float32 | True |
| blocks.1.1.bn3.weight | parameter | [24] | float32 | True |
| blocks.1.1.bn3.bias | parameter | [24] | float32 | True |
| blocks.1.1.bn3.running_mean | buffer | [24] | float32 | True |
| blocks.1.1.bn3.running_var | buffer | [24] | float32 | True |
| blocks.1.1.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.2.0.conv_pw.weight | parameter | [96, 24, 1, 1] | float32 | True |
| blocks.2.0.bn1.weight | parameter | [96] | float32 | True |
| blocks.2.0.bn1.bias | parameter | [96] | float32 | True |
| blocks.2.0.bn1.running_mean | buffer | [96] | float32 | True |
| blocks.2.0.bn1.running_var | buffer | [96] | float32 | True |
| blocks.2.0.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.2.0.conv_dw.weight | parameter | [96, 1, 5, 5] | float32 | True |
| blocks.2.0.bn2.weight | parameter | [96] | float32 | True |
| blocks.2.0.bn2.bias | parameter | [96] | float32 | True |
| blocks.2.0.bn2.running_mean | buffer | [96] | float32 | True |
| blocks.2.0.bn2.running_var | buffer | [96] | float32 | True |
| blocks.2.0.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.2.0.se.conv_reduce.weight | parameter | [24, 96, 1, 1] | float32 | True |
| blocks.2.0.se.conv_reduce.bias | parameter | [24] | float32 | True |
| blocks.2.0.se.conv_expand.weight | parameter | [96, 24, 1, 1] | float32 | True |
| blocks.2.0.se.conv_expand.bias | parameter | [96] | float32 | True |
| blocks.2.0.conv_pwl.weight | parameter | [40, 96, 1, 1] | float32 | True |
| blocks.2.0.bn3.weight | parameter | [40] | float32 | True |
| blocks.2.0.bn3.bias | parameter | [40] | float32 | True |
| blocks.2.0.bn3.running_mean | buffer | [40] | float32 | True |
| blocks.2.0.bn3.running_var | buffer | [40] | float32 | True |
| blocks.2.0.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.2.1.conv_pw.weight | parameter | [240, 40, 1, 1] | float32 | True |
| blocks.2.1.bn1.weight | parameter | [240] | float32 | True |
| blocks.2.1.bn1.bias | parameter | [240] | float32 | True |
| blocks.2.1.bn1.running_mean | buffer | [240] | float32 | True |
| blocks.2.1.bn1.running_var | buffer | [240] | float32 | True |
| blocks.2.1.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.2.1.conv_dw.weight | parameter | [240, 1, 5, 5] | float32 | True |
| blocks.2.1.bn2.weight | parameter | [240] | float32 | True |
| blocks.2.1.bn2.bias | parameter | [240] | float32 | True |
| blocks.2.1.bn2.running_mean | buffer | [240] | float32 | True |
| blocks.2.1.bn2.running_var | buffer | [240] | float32 | True |
| blocks.2.1.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.2.1.se.conv_reduce.weight | parameter | [64, 240, 1, 1] | float32 | True |
| blocks.2.1.se.conv_reduce.bias | parameter | [64] | float32 | True |
| blocks.2.1.se.conv_expand.weight | parameter | [240, 64, 1, 1] | float32 | True |
| blocks.2.1.se.conv_expand.bias | parameter | [240] | float32 | True |
| blocks.2.1.conv_pwl.weight | parameter | [40, 240, 1, 1] | float32 | True |
| blocks.2.1.bn3.weight | parameter | [40] | float32 | True |
| blocks.2.1.bn3.bias | parameter | [40] | float32 | True |
| blocks.2.1.bn3.running_mean | buffer | [40] | float32 | True |
| blocks.2.1.bn3.running_var | buffer | [40] | float32 | True |
| blocks.2.1.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.2.2.conv_pw.weight | parameter | [240, 40, 1, 1] | float32 | True |
| blocks.2.2.bn1.weight | parameter | [240] | float32 | True |
| blocks.2.2.bn1.bias | parameter | [240] | float32 | True |
| blocks.2.2.bn1.running_mean | buffer | [240] | float32 | True |
| blocks.2.2.bn1.running_var | buffer | [240] | float32 | True |
| blocks.2.2.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.2.2.conv_dw.weight | parameter | [240, 1, 5, 5] | float32 | True |
| blocks.2.2.bn2.weight | parameter | [240] | float32 | True |
| blocks.2.2.bn2.bias | parameter | [240] | float32 | True |
| blocks.2.2.bn2.running_mean | buffer | [240] | float32 | True |
| blocks.2.2.bn2.running_var | buffer | [240] | float32 | True |
| blocks.2.2.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.2.2.se.conv_reduce.weight | parameter | [64, 240, 1, 1] | float32 | True |
| blocks.2.2.se.conv_reduce.bias | parameter | [64] | float32 | True |
| blocks.2.2.se.conv_expand.weight | parameter | [240, 64, 1, 1] | float32 | True |
| blocks.2.2.se.conv_expand.bias | parameter | [240] | float32 | True |
| blocks.2.2.conv_pwl.weight | parameter | [40, 240, 1, 1] | float32 | True |
| blocks.2.2.bn3.weight | parameter | [40] | float32 | True |
| blocks.2.2.bn3.bias | parameter | [40] | float32 | True |
| blocks.2.2.bn3.running_mean | buffer | [40] | float32 | True |
| blocks.2.2.bn3.running_var | buffer | [40] | float32 | True |
| blocks.2.2.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.3.0.conv_pw.weight | parameter | [120, 40, 1, 1] | float32 | True |
| blocks.3.0.bn1.weight | parameter | [120] | float32 | True |
| blocks.3.0.bn1.bias | parameter | [120] | float32 | True |
| blocks.3.0.bn1.running_mean | buffer | [120] | float32 | True |
| blocks.3.0.bn1.running_var | buffer | [120] | float32 | True |
| blocks.3.0.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.3.0.conv_dw.weight | parameter | [120, 1, 5, 5] | float32 | True |
| blocks.3.0.bn2.weight | parameter | [120] | float32 | True |
| blocks.3.0.bn2.bias | parameter | [120] | float32 | True |
| blocks.3.0.bn2.running_mean | buffer | [120] | float32 | True |
| blocks.3.0.bn2.running_var | buffer | [120] | float32 | True |
| blocks.3.0.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.3.0.se.conv_reduce.weight | parameter | [32, 120, 1, 1] | float32 | True |
| blocks.3.0.se.conv_reduce.bias | parameter | [32] | float32 | True |
| blocks.3.0.se.conv_expand.weight | parameter | [120, 32, 1, 1] | float32 | True |
| blocks.3.0.se.conv_expand.bias | parameter | [120] | float32 | True |
| blocks.3.0.conv_pwl.weight | parameter | [48, 120, 1, 1] | float32 | True |
| blocks.3.0.bn3.weight | parameter | [48] | float32 | True |
| blocks.3.0.bn3.bias | parameter | [48] | float32 | True |
| blocks.3.0.bn3.running_mean | buffer | [48] | float32 | True |
| blocks.3.0.bn3.running_var | buffer | [48] | float32 | True |
| blocks.3.0.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.3.1.conv_pw.weight | parameter | [144, 48, 1, 1] | float32 | True |
| blocks.3.1.bn1.weight | parameter | [144] | float32 | True |
| blocks.3.1.bn1.bias | parameter | [144] | float32 | True |
| blocks.3.1.bn1.running_mean | buffer | [144] | float32 | True |
| blocks.3.1.bn1.running_var | buffer | [144] | float32 | True |
| blocks.3.1.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.3.1.conv_dw.weight | parameter | [144, 1, 5, 5] | float32 | True |
| blocks.3.1.bn2.weight | parameter | [144] | float32 | True |
| blocks.3.1.bn2.bias | parameter | [144] | float32 | True |
| blocks.3.1.bn2.running_mean | buffer | [144] | float32 | True |
| blocks.3.1.bn2.running_var | buffer | [144] | float32 | True |
| blocks.3.1.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.3.1.se.conv_reduce.weight | parameter | [40, 144, 1, 1] | float32 | True |
| blocks.3.1.se.conv_reduce.bias | parameter | [40] | float32 | True |
| blocks.3.1.se.conv_expand.weight | parameter | [144, 40, 1, 1] | float32 | True |
| blocks.3.1.se.conv_expand.bias | parameter | [144] | float32 | True |
| blocks.3.1.conv_pwl.weight | parameter | [48, 144, 1, 1] | float32 | True |
| blocks.3.1.bn3.weight | parameter | [48] | float32 | True |
| blocks.3.1.bn3.bias | parameter | [48] | float32 | True |
| blocks.3.1.bn3.running_mean | buffer | [48] | float32 | True |
| blocks.3.1.bn3.running_var | buffer | [48] | float32 | True |
| blocks.3.1.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.4.0.conv_pw.weight | parameter | [288, 48, 1, 1] | float32 | True |
| blocks.4.0.bn1.weight | parameter | [288] | float32 | True |
| blocks.4.0.bn1.bias | parameter | [288] | float32 | True |
| blocks.4.0.bn1.running_mean | buffer | [288] | float32 | True |
| blocks.4.0.bn1.running_var | buffer | [288] | float32 | True |
| blocks.4.0.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.4.0.conv_dw.weight | parameter | [288, 1, 5, 5] | float32 | True |
| blocks.4.0.bn2.weight | parameter | [288] | float32 | True |
| blocks.4.0.bn2.bias | parameter | [288] | float32 | True |
| blocks.4.0.bn2.running_mean | buffer | [288] | float32 | True |
| blocks.4.0.bn2.running_var | buffer | [288] | float32 | True |
| blocks.4.0.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.4.0.se.conv_reduce.weight | parameter | [72, 288, 1, 1] | float32 | True |
| blocks.4.0.se.conv_reduce.bias | parameter | [72] | float32 | True |
| blocks.4.0.se.conv_expand.weight | parameter | [288, 72, 1, 1] | float32 | True |
| blocks.4.0.se.conv_expand.bias | parameter | [288] | float32 | True |
| blocks.4.0.conv_pwl.weight | parameter | [96, 288, 1, 1] | float32 | True |
| blocks.4.0.bn3.weight | parameter | [96] | float32 | True |
| blocks.4.0.bn3.bias | parameter | [96] | float32 | True |
| blocks.4.0.bn3.running_mean | buffer | [96] | float32 | True |
| blocks.4.0.bn3.running_var | buffer | [96] | float32 | True |
| blocks.4.0.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.4.1.conv_pw.weight | parameter | [576, 96, 1, 1] | float32 | True |
| blocks.4.1.bn1.weight | parameter | [576] | float32 | True |
| blocks.4.1.bn1.bias | parameter | [576] | float32 | True |
| blocks.4.1.bn1.running_mean | buffer | [576] | float32 | True |
| blocks.4.1.bn1.running_var | buffer | [576] | float32 | True |
| blocks.4.1.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.4.1.conv_dw.weight | parameter | [576, 1, 5, 5] | float32 | True |
| blocks.4.1.bn2.weight | parameter | [576] | float32 | True |
| blocks.4.1.bn2.bias | parameter | [576] | float32 | True |
| blocks.4.1.bn2.running_mean | buffer | [576] | float32 | True |
| blocks.4.1.bn2.running_var | buffer | [576] | float32 | True |
| blocks.4.1.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.4.1.se.conv_reduce.weight | parameter | [144, 576, 1, 1] | float32 | True |
| blocks.4.1.se.conv_reduce.bias | parameter | [144] | float32 | True |
| blocks.4.1.se.conv_expand.weight | parameter | [576, 144, 1, 1] | float32 | True |
| blocks.4.1.se.conv_expand.bias | parameter | [576] | float32 | True |
| blocks.4.1.conv_pwl.weight | parameter | [96, 576, 1, 1] | float32 | True |
| blocks.4.1.bn3.weight | parameter | [96] | float32 | True |
| blocks.4.1.bn3.bias | parameter | [96] | float32 | True |
| blocks.4.1.bn3.running_mean | buffer | [96] | float32 | True |
| blocks.4.1.bn3.running_var | buffer | [96] | float32 | True |
| blocks.4.1.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.4.2.conv_pw.weight | parameter | [576, 96, 1, 1] | float32 | True |
| blocks.4.2.bn1.weight | parameter | [576] | float32 | True |
| blocks.4.2.bn1.bias | parameter | [576] | float32 | True |
| blocks.4.2.bn1.running_mean | buffer | [576] | float32 | True |
| blocks.4.2.bn1.running_var | buffer | [576] | float32 | True |
| blocks.4.2.bn1.num_batches_tracked | buffer | [] | int64 | True |
| blocks.4.2.conv_dw.weight | parameter | [576, 1, 5, 5] | float32 | True |
| blocks.4.2.bn2.weight | parameter | [576] | float32 | True |
| blocks.4.2.bn2.bias | parameter | [576] | float32 | True |
| blocks.4.2.bn2.running_mean | buffer | [576] | float32 | True |
| blocks.4.2.bn2.running_var | buffer | [576] | float32 | True |
| blocks.4.2.bn2.num_batches_tracked | buffer | [] | int64 | True |
| blocks.4.2.se.conv_reduce.weight | parameter | [144, 576, 1, 1] | float32 | True |
| blocks.4.2.se.conv_reduce.bias | parameter | [144] | float32 | True |
| blocks.4.2.se.conv_expand.weight | parameter | [576, 144, 1, 1] | float32 | True |
| blocks.4.2.se.conv_expand.bias | parameter | [576] | float32 | True |
| blocks.4.2.conv_pwl.weight | parameter | [96, 576, 1, 1] | float32 | True |
| blocks.4.2.bn3.weight | parameter | [96] | float32 | True |
| blocks.4.2.bn3.bias | parameter | [96] | float32 | True |
| blocks.4.2.bn3.running_mean | buffer | [96] | float32 | True |
| blocks.4.2.bn3.running_var | buffer | [96] | float32 | True |
| blocks.4.2.bn3.num_batches_tracked | buffer | [] | int64 | True |
| blocks.5.0.conv.weight | parameter | [576, 96, 1, 1] | float32 | True |
| blocks.5.0.bn1.weight | parameter | [576] | float32 | True |
| blocks.5.0.bn1.bias | parameter | [576] | float32 | True |
| blocks.5.0.bn1.running_mean | buffer | [576] | float32 | True |
| blocks.5.0.bn1.running_var | buffer | [576] | float32 | True |
| blocks.5.0.bn1.num_batches_tracked | buffer | [] | int64 | True |
| conv_head.weight | parameter | [1024, 576, 1, 1] | float32 | True |
| conv_head.bias | parameter | [1024] | float32 | True |
| classifier.weight | parameter | [1000, 1024] | float32 | True |
| classifier.bias | parameter | [1000] | float32 | True |
