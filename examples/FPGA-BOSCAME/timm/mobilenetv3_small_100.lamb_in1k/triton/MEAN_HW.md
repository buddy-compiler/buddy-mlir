# mean_hw：7 个静态 Triton reduction cases

> 2026-10-06 后续上板结果见 [逐 case 验证记录](../validation/fpga-20261006/README.md)。下文的 `NOT EXECUTED` 为初次构建时的历史状态；patch5 与 patch6 的结果分别记录。

实现位于此前合并后的 `examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/`。
本次仅新增一个通用 Triton kernel：`kernels.py::mean_hw`。
模型 profile 保持 batch=1、输入 `[1,3,224,224]`、NCHW、FP32、eval、输出 `[1,1000]`。

## 语义与真实 reduction

输入为连续 `[1,C,H,W]`，输出为 `[1,C,1,1]`，等价于
`aten.mean.dim(X, [2,3], keepdim=True)`：

```text
Out[c] = sum(X[0,c,h,w] for h,w) / (H*W)
```

每个 program 处理一个通道，grid 为 `[C,1,1]`，输入地址为
`X + channel*(H*W) + arange(0,BLOCK)`。BLOCK 是不小于 H*W 的最小 2 次幂。
`spatial < H*W` 的 mask 将无效 lane 置零；`tl.sum(x, axis=0)` 在 Triton 内进行
真正的 FP32 归约，然后 `tl.div_rn(total, H*W)`，最终只向 `Out[channel]` 写一个 FP32。
分母始终为真实 H*W。Out 的有效存储是 C 个元素，保持维度由静态 output_shape 明确记录。

| case | Input NCHW | Output NCHW | H*W | BLOCK | masked lane | grid |
| --- | --- | --- | ---: | ---: | ---: | --- |
| mean_hw_c16_h56_w56 | [1,16,56,56] | [1,16,1,1] | 3136 | 4096 | 960 | [16,1,1] |
| mean_hw_c96_h14_w14 | [1,96,14,14] | [1,96,1,1] | 196 | 256 | 60 | [96,1,1] |
| mean_hw_c240_h14_w14 | [1,240,14,14] | [1,240,1,1] | 196 | 256 | 60 | [240,1,1] |
| mean_hw_c120_h14_w14 | [1,120,14,14] | [1,120,1,1] | 196 | 256 | 60 | [120,1,1] |
| mean_hw_c144_h14_w14 | [1,144,14,14] | [1,144,1,1] | 196 | 256 | 60 | [144,1,1] |
| mean_hw_c288_h7_w7 | [1,288,7,7] | [1,288,1,1] | 49 | 64 | 15 | [288,1,1] |
| mean_hw_c576_h7_w7 | [1,576,7,7] | [1,576,1,1] | 49 | 64 | 15 | [576,1,1] |

清单覆盖 `model.json` 中全部 10 次 `aten.mean.dim` 调用（9 次 SE、1 次 global pool）
的 7 组去重输入/输出 shape，dim 为 `[2,3]` 或等价的 `[-1,-2]`，keepdim 均为 true。
本类按 `(C,H,W)` specialization，`(144,14,14)` 和 `(576,7,7)` 的输入 COUNT
虽相同，reduction width 和输出长度均不同，因此保留两个 case。
没有卷积权重、OIHW 或空间 padding；这里的补零仅用于非 2 次幂 reduction 的 mask。

`cases.py --family=mean_hw --generate` 从统一 inventory 与 `launch_mean_hw.c.in`
自动生成全部 7 个目录，各含 `launch.c`、`metadata.json`、`makefile`，正常 build 也自动生成。

## 独立 C reference 与误差容限

每个 `launch.c` 在 kernel 调用前，以独立的 NCHW `c/h/w` 三重循环，
将输入 FP32 转为 double 后累加，最后用 double 除以 H*W。
生产 kernel 的累加和除法仍全部为 FP32，没有使用 C 或 FP64 替代 Triton 计算。

容限按每个通道独立计算。令 `S=H*W`，FP32 unit roundoff 为 `u=2^-24`，
`A_c = sum(abs(X[c,h,w]))/S`，定义：

```text
gamma_k = (k*u)/(1-k*u)
T_c = gamma_(S+1) * A_c + 2^-149
abs(Out[c] - double_reference[c]) <= T_c
```

依据是浮点求和的前向误差界 `gamma_(S-1)*sum(abs(x))`，见
[Higham, The Accuracy of Floating Point Summation, 式 (2.6)](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf)。
加入最后一次 FP32 除法后可用 `gamma_S*A_c`；这里额外计入一次 FP32 舍入，
覆盖 double reference、double 绝对值和及容限求值的误差余量。
这些宽度下的 FP64 累加误差系数小于 `3.5e-13`，远低于额外 FP32 项约 `5.96e-8`。
`2^-149` 是一个最小 FP32 subnormal 的绝对余量，也使全零通道的容限有定义。

这个界不假设平衡树归约，适用于当前 Linalg lowering 的顺序 FP32 累加。
masked lane 加零是精确操作，不按补齐后的 BLOCK 放大容限。
测试输入全部有限，并保证 FP32 中间和不溢出；误差依据采用 round-to-nearest。
使用 `mean(abs(X))` 而非 `abs(mean(X))`，可以正确处理正负抵消、均值接近零的情况。

| S | gamma_(S+1) |
| ---: | ---: |
| 3136 | 1.8701473863334017e-4 |
| 196 | 1.1742252899636104e-5 |
| 49 | 2.980241120580198e-6 |

系数在测试运行前由形状和 FP32 精度确定，没有根据观测误差放宽阈值。
构建脚本对 mean_hw 读取真实非零误差和 `max_error_over_bound`，要求后者不超过 1；
此前各类 elementwise 的零误差验收要求保持不变。

## 测试覆盖及实际结果

C oracle 有 7 轮输入：正负零、各通道不同的常量、末尾脉冲、非二进制精确的有符号值、
正负抵消、全正的小增量序列、开头脉冲。首尾脉冲检查 mask、NCHW 通道边界及正确分母。
Host 在 `X[COUNT]` 和 `Out[C]` 后设置 PROT_NONE 页，前方有 canary；
NR 缓冲区前后各有 128 个 guard。输入不可变性逐元素按位检查。

`verify_mean_hw.py` 将同一 Host LLVM IR 与 ABI adapter 链接为动态库，
调用真实编译产物，另外对照 FP64 mean 和 `torch.ops.aten.mean.dim(X,[2,3],True)`。
每个 case 有 seed=0 的 20 轮输入，含布局/边界/抵消测试及 `2^-10`、1、`2^10`
三种幅度的随机输入。两种 FP32 输出各自必须在 FP64 reference 的 T_c 内；
它们之间的差异还必须满足三角不等式给出的 `2*T_c`。
此检查补充独立 C oracle，不用 Triton 输出生成 reference。

| case | C double reference 最大绝对误差 | C 最大 error/bound |
| --- | ---: | ---: |
| mean_hw_c16_h56_w56 | 3.1609848445413036e-6 | 0.01247612 |
| mean_hw_c96_h14_w14 | 1.7698930232512566e-7 | 0.01254559 |
| mean_hw_c240_h14_w14 | 1.7698930232512566e-7 | 0.01254559 |
| mean_hw_c120_h14_w14 | 1.7698930232512566e-7 | 0.01254559 |
| mean_hw_c144_h14_w14 | 1.7698930232512566e-7 | 0.01254559 |
| mean_hw_c288_h7_w7 | 1.2894066014901284e-7 | 0.03087588 |
| mean_hw_c576_h7_w7 | 1.2894066014901284e-7 | 0.03087588 |

- 独立 C Host 检查：**7/7 PASS**。
- PyTorch 2.10.0+cpu / FP64 对照：**7/7 PASS**，共 **29,600** 个输出、**4,202,240** 个输入样本。
  多幅度测试对 FP64 的最大绝对误差为 **6.427375637940713e-4**，
  对 aten FP32 的最大差异为 **8.544921875e-4**；最大 error/bound 为 **0.05804058**。
  绝对误差随输入幅度变化，所有逐通道判断均在预先规定的容限内。
- NR ELF/BIN 构建及 ELF 指令审计：**7/7 PASS**。
- 已有算子 Host 回归：**38/38 PASS**，原有 inventory、kernel 语义、ABI adapter 不变。
- FPGA：**NOT EXECUTED**，沿用用户暂停上板的约定。
- 本次状态：**COMPLETE**，范围为 mean_hw 实现、Host 验证和 NR 二进制构建。

证据：[host.json](../validation/mean-hw/host.json)、[nr.json](../validation/mean-hw/nr.json)、
[regression.json](../validation/mean-hw/regression.json)、[status.json](../validation/mean-hw/status.json)。
回归产物位于 `triton/build/regression-mean-hw/`，此前各类的产物和报告保留。

## 实际构建链与命令

Triton AST → TTIR（`tt.reduce`）→ triton-riscv `triton-to-linalg-experimental`
→ Linalg（`linalg.reduce`、`arith.addf`、`arith.divf`）→ Buddy bufferization/lowering
→ LLVM → RISC-V（`fadd.s`、`fdiv.s`）→ NR ELF/BIN。
导出工具对 mean_hw 明确要求真实 `linalg.reduce`。没有手写 `kernel.mlir`。
Qwen3、公共 runtime 和上板脚本未修改。

仓库根目录运行本机已验证的命令：

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
MOBILENET_PYTHON=/home/zhangwenji/triton-riscv/.venv/bin/python
MOBILENET_TRITON=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton

"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=mean_hw --host
"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=mean_hw --nr
```

`--nr` 在全部 7 个 Host 验收通过后才开始 NR 编译；仅当源码、工具和产物摘要一致时
复用已有 PASS 证据。也支持以下 Make 入口：

```bash
make -C "$MOBILENET_TRITON" FAMILY=mean_hw check TRITON_PYTHON="$MOBILENET_PYTHON"
make -C "$MOBILENET_TRITON" FAMILY=mean_hw all TRITON_PYTHON="$MOBILENET_PYTHON"
```

使用本机安装的外部 Triton/Buddy/LLVM。实际路径、版本、SHA-256 见
[toolchain.json](../validation/mean-hw/toolchain.json)，不声称匹配 Qwen3 的锁定工具链版本。

## 产物及后续上板

每个 case 的产物均位于 `triton/build/<case>/`：

| 产物 | 相对路径 |
| --- | --- |
| TTIR | `kernel.ttir` |
| Linalg MLIR | `kernel.linalg.mlir` |
| 前端记录 / grid ABI adapter | `frontend.json`、`adapter.c` |
| Host LLVM / C oracle / 日志 | `host/kernel.ll`、`host/check`、`host/output.log` |
| FP64/ATen 对照库 / 报告 | `host/pytorch-check.so`、`host/pytorch.json` |
| NR LLVM dialect / LLVM IR | `nr/kernel.llvm.mlir`、`nr/kernel.ll` |
| RISC-V 汇编 | `nr/kernel.nr.S` |
| ELF / BIN | `nr/<case>.elf`、`nr/<case>.bin` |
| ELF 审计 | `nr/elf-audit.json` |

例如最大 reduction width 的 `mean_hw_c16_h56_w56`：
`triton/build/mean_hw_c16_h56_w56/nr/mean_hw_c16_h56_w56.bin`。
以下命令供后续使用，**本次未执行**：

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton/build/mean_hw_c16_h56_w56/nr/mean_hw_c16_h56_w56.bin \
  --fpga=5 --capture-seconds=120 --completion-marker='[nr] RA returned:'
```

`build.py --family=mean_hw --run` 可依次调用同一个公共脚本。脚本必须成功退出，
并实际出现对应 case 的数值 PASS 和 `[nr] RA returned: PASS`，才记录 FPGA PASS。
mean_hw 的 PASS 允许容限内的非零数值误差；本次未连接服务器或执行板上程序。
