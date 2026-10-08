# MobileNetV3 NCHW depthwise_conv2d

> 2026-10-06 后续上板结果见 [逐 case 验证记录](../validation/fpga-20261006/README.md)。下文的 `NOT EXECUTED` 为初次构建时的历史状态；patch5 与 patch6 的结果分别记录。

本次在已合并的 `examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/`
目录新增一个通用 Triton kernel：`kernels.py::depthwise_conv2d`，通过
`cases.py` 的统一 inventory 自动生成以下 9 个 constexpr specialization。
固定 N=1、FP32、NCHW contiguous、groups=C；权重物理布局是 OIHW `[C,1,KH,KW]`。
Bias[C] 作为必需输入，接收离线 BN folding 后的 bias。

| case | 输入 [C,H,W] | K / stride / pad | 输出 [C,OH,OW] | grid | 最后 tile 有效 lane |
| --- | --- | --- | --- | --- | ---: |
| dwconv_c16_h112_k3_s2_p1 | [16,112,112] | 3 / 2 / 1 | [16,56,56] | [25,16,1] | 64 |
| dwconv_c72_h56_k3_s2_p1 | [72,56,56] | 3 / 2 / 1 | [72,28,28] | [7,72,1] | 16 |
| dwconv_c88_h28_k3_s1_p1 | [88,28,28] | 3 / 1 / 1 | [88,28,28] | [7,88,1] | 16 |
| dwconv_c96_h28_k5_s2_p2 | [96,28,28] | 5 / 2 / 2 | [96,14,14] | [2,96,1] | 68 |
| dwconv_c240_h14_k5_s1_p2 | [240,14,14] | 5 / 1 / 2 | [240,14,14] | [2,240,1] | 68 |
| dwconv_c120_h14_k5_s1_p2 | [120,14,14] | 5 / 1 / 2 | [120,14,14] | [2,120,1] | 68 |
| dwconv_c144_h14_k5_s1_p2 | [144,14,14] | 5 / 1 / 2 | [144,14,14] | [2,144,1] | 68 |
| dwconv_c288_h14_k5_s2_p2 | [288,14,14] | 5 / 2 / 2 | [288,7,7] | [1,288,1] | 49 |
| dwconv_c576_h7_k5_s1_p2 | [576,7,7] | 5 / 1 / 2 | [576,7,7] | [1,576,1] | 49 |

这 9 组配置覆盖本地 `model.json` 中全部 11 次 depthwise 调用，
C=240/H=14 和 C=576/H=7 各复用两次。实际模块匹配记录见
[`model-coverage.json`](../validation/depthwise-conv2d/model-coverage.json)。
模型整体 profile 保持输入 `[1,3,224,224]`、输出 `[1,1000]`、eval；本次验证的是这类独立算子。

## Triton 实现与真实编译链

BLOCK=128。`program_id(1)` 选择一个 channel，`program_id(0)` 选择该 channel
的输出空间 tile。lane 的扁平 spatial index 按 `oh=spatial//OW`、`ow=spatial%OW`
还原；从 `ih=oh*SH+kh-PH`、`iw=ow*SW+kw-PW` 计算采样位置。

每个 `kh,kw` 都独立检查 `0<=ih<H`、`0<=iw<W` 以及输出 tail mask。
有效值直接从 `X[(c*H+ih)*W+iw]` 读取，padding 返回 FP32 零；权重从
`Weight[(c*KH+kh)*KW+kw]` 读取。每个 lane 在 Triton 内累加 9 或 25 个 FP32
乘积，最后加 Bias[c]，用输出 mask 写到 `Out[c*OH*OW+spatial]`。
不跨 channel 累加，不预先展开 im2col，不调用 dense GEMM，不翻转卷积核。

所有数值计算均在这一份 Triton JIT 函数中；C adapter 只处理 descriptor 和 grid。
真实生成的 TTIR 分别含 9/25 个 `arith.mulf` 和对应累加，mask 经 triton-riscv
降为带条件 load 的 Linalg/SCF。exporter 拒绝 depthwise 中出现 `tt.dot` 或 Linalg matmul。

编译为：Triton ASTSource → TTIR → triton-riscv Linalg → Buddy bufferization / loops
→ LLVM dialect → LLVM IR → RISC-V assembly → 共享 NR runtime 链接 → ELF/BIN。
没有手写 `kernel.mlir`，也没有用 C oracle 替代 kernel。

当前 NR 沿用本目录已有的 scalar FP32 lowering；Triton 的 grid 按 channel/spatial
划分独立工作，现有 NR adapter 顺序提交 program，不宣称实际 FPGA 并行加速。
9 份 NR LLVM IR 与对应已执行的 Host LLVM IR 逐字节相同。
这证明检查了同一份 LLVM 计算程序，不能代替 RISC-V/FPGA 实际运行。

## 独立 oracle、边界和误差

每个 case 都有自动生成的 `launch.c`，独立 C `c/oh/ow/kh/kw` 五重循环用 double
计算 reference，显式跳过四侧 padding，直接读取 NCHW/OIHW，再加入 Bias[c]。
8 轮输入包含：bias-only 和 ±0、空间/通道 ramp、四角脉冲、仅最后 channel 非零、
右上/左下两个非对称 one-hot 卷积核、棋盘格抵消，以及有正负值的非二进制精确输入。

四个 Host buffer 的有效末尾分别紧邻 PROT_NONE 页面；检查前缀 canary，以及
X/Weight/Bias 逐位不变。NR 使用同一套 C oracle 和前后 canary。
输出尾部、padding 行列与相邻 channel 不会被当作有效数据。

令 L=KH*KW，u=2^-24，使用每个输出的前向误差界：

```text
A[c,oh,ow] = sum_valid(abs(double(X)*double(Weight))) + abs(double(Bias[c]))
gamma = ((L+2)*u) / (1-(L+2)*u)
bound = gamma*A + (L+2)*2^-149
abs(actual - double_reference) <= bound
```

一个乘积最多经历一次 FP32 乘法、L-1 次加法和最后的 bias 加法；额外一次 FP32
舍入余量覆盖 double reference 和容限求值误差。3×3 的 gamma 为
`6.556515224079339e-7`，5×5 为 `1.6093279988679868e-6`。
用绝对值乘积之和保留抵消情况下的有效误差尺度。系数在测试前由形状确定，
没有按测到的误差放宽。误差分析针对有限输入、round-to-nearest 和无中间溢出的计算。

`verify_depthwise_conv2d.py` 另外对实际编译的 LLVM 执行每 case 12 轮固定 seed=0
的 FP64 grouped conv2d / `aten.convolution.default` 对照，含 2^-4 / 1 / 2^4
尺度的随机输入与权重。两个 FP32 结果分别满足 FP64 界，彼此差异不超过两倍界。
它还单独逆序调用首/中/末 channel 的所有 spatial program，检查 sentinel 输出中
只有该 program 负责的区间发生写入。总计 147 次独立 program 检查通过。

| case | C max_abs_error | C max(error/bound) | FP64/ATen 压力测试 max_abs_error | Host / NR build |
| --- | ---: | ---: | ---: | --- |
| dwconv_c16_h112_k3_s2_p1 | 6.914846811e-7 | 0.274263 | 5.952579177e-4 | PASS / PASS |
| dwconv_c72_h56_k3_s2_p1 | 1.668930054e-6 | 0.274263 | 5.637002755e-4 | PASS / PASS |
| dwconv_c88_h28_k3_s1_p1 | 1.996755600e-6 | 0.308498 | 5.723947143e-4 | PASS / PASS |
| dwconv_c96_h28_k5_s2_p2 | 2.294778824e-6 | 0.104252 | 1.023918831e-3 | PASS / PASS |
| dwconv_c240_h14_k5_s1_p2 | 7.361173630e-6 | 0.134591 | 1.124560282e-3 | PASS / PASS |
| dwconv_c120_h14_k5_s1_p2 | 3.591179848e-6 | 0.134591 | 1.451069956e-3 | PASS / PASS |
| dwconv_c144_h14_k5_s1_p2 | 3.620982170e-6 | 0.134591 | 1.164560926e-3 | PASS / PASS |
| dwconv_c288_h14_k5_s2_p2 | 7.025897503e-6 | 0.110553 | 1.002732896e-3 | PASS / PASS |
| dwconv_c576_h7_k5_s1_p2 | 1.493841410e-5 | 0.138940 | 1.151387140e-3 | PASS / PASS |

压力测试列也是相对 FP64 reference 的误差，输入/权重尺度与 C 测试不同；
其最大 error/bound 为 0.437610，全部小于 1。
原有 46 个 case 的 Host 回归全部 PASS；原 kernel AST、inventory 和生成的 adapter 不变。

## 复现与上板命令

仓库根目录运行，以下是本次实际使用的本机工具：

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
P=/home/zhangwenji/triton-riscv/.venv/bin/python
M=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k

# 自动生成 9 个目录；正常 build 也会自动生成。
"$P" "$M/triton/cases.py" --family=depthwise_conv2d --generate

# 9 个独立 C oracle + FP64/ATen + program 写入边界检查。
"$P" "$M/triton/build.py" --family=depthwise_conv2d --host

# 在本 family 全部 Host PASS 后才构建任何一个 NR 镜像。
"$P" "$M/triton/build.py" --family=depthwise_conv2d --nr
```

make 入口为 `make -C "$M/triton" FAMILY=depthwise_conv2d TRITON_PYTHON="$P" check`
和 `all`。真实工具路径、版本、参数、SHA-256 在 `validation/depthwise-conv2d/toolchain.json`。
使用的是本机已安装的外部 compiler build，未声称重建了本仓库 pinned toolchain。

此前要求暂停上板、保留本地产物，本次未尝试 SSH、scp 或 FPGA 运行。
未来运行任一 case 时复用原脚本，例如：

```bash
CASE=dwconv_c16_h112_k3_s2_p1
examples/FPGA-BOSCAME/fpga_run.sh \
  "$M/triton/build/$CASE/nr/$CASE.bin" \
  --fpga=5 --capture-seconds=120 --completion-marker='[nr] RA returned:'
```

运行全部 case 可用 `"$P" "$M/triton/build.py" --family=depthwise_conv2d --run`，
仍只调用同一个现有脚本。任一上板 case 未通过就停止；只有捕获对应数值 PASS、
`[nr] RA returned: PASS` 且脚本成功退出，才记录 FPGA PASS。

## 文件和产物

修改：`triton/kernels.py`、`triton/cases.py`、`triton/export.py`、`triton/build.py`、
根 `README.md` 和 `triton/README.md`。
新增：`triton/launch_depthwise_conv2d.c.in`、`triton/verify_depthwise_conv2d.py`、
本文档，以及 9 个 case 目录各自的 `launch.c`、`metadata.json`、`makefile`。
没有修改 Qwen、`common.mk`、`support.c/h`、共享 runtime 或 FPGA 运行脚本。

全部 case 的 ELF/BIN 路径统一为：

```text
$M/triton/build/<case>/nr/<case>.elf
$M/triton/build/<case>/nr/<case>.bin
```

| case | ELF bytes | BIN bytes |
| --- | ---: | ---: |
| dwconv_c16_h112_k3_s2_p1 | 43,376 | 82,207 |
| dwconv_c72_h56_k3_s2_p1 | 43,344 | 82,366 |
| dwconv_c88_h28_k3_s1_p1 | 43,272 | 82,294 |
| dwconv_c96_h28_k5_s2_p2 | 52,992 | 89,742 |
| dwconv_c240_h14_k5_s1_p2 | 52,888 | 90,007 |
| dwconv_c120_h14_k5_s1_p2 | 52,888 | 90,007 |
| dwconv_c144_h14_k5_s1_p2 | 52,888 | 90,007 |
| dwconv_c288_h14_k5_s2_p2 | 52,960 | 89,855 |
| dwconv_c576_h7_k5_s1_p2 | 52,856 | 89,982 |

以 `B=$M/triton/build/dwconv_c16_h112_k3_s2_p1` 为例：

| 产物 | 路径 |
| --- | --- |
| TTIR | `$B/kernel.ttir` |
| triton-riscv Linalg MLIR | `$B/kernel.linalg.mlir` |
| Host LLVM IR | `$B/host/kernel.ll` |
| NR LLVM dialect / LLVM IR | `$B/nr/kernel.llvm.mlir` / `$B/nr/kernel.ll` |
| NR assembly | `$B/nr/kernel.nr.S` |
| NR BIN | `$B/nr/dwconv_c16_h112_k3_s2_p1.bin` |
| Host numerical log | `$B/host/output.log` |
| ELF 审计 | `$B/nr/elf-audit.json` |

证据在 `validation/depthwise-conv2d/{host,nr,regression,model-coverage,toolchain,status}.json`，
包含逐 case 数值结果、编译器/源文件/产物 SHA-256、NR ELF 审计和实际路径。

本地实现、Host 验证和 NR 构建：**COMPLETE**。
全部 9 个 case 的 FPGA 实际结果：**NOT EXECUTED**；尚未完成板上数值验证。
