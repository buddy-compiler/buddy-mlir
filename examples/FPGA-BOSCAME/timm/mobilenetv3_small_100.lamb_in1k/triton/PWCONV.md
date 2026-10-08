# NCHW FP32 pointwise_conv2d

> 2026-10-06 后续上板结果见 [逐 case 验证记录](../validation/fpga-20261006/README.md)。下文的 `NOT EXECUTED` 为初次构建时的历史状态；patch5 与 patch6 的结果分别记录。

本次实现位于此前合并后的 `examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/`。
一个通用 Triton kernel：`kernels.py::pointwise_conv2d`，32 个静态 specialization，
统一由 `cases.py` inventory 与 `launch_pointwise_conv2d.c.in` 自动生成目录。

固定 batch=1、FP32、1×1、stride=1、padding=0、groups=1。
Input/Output 均为连续 NCHW，Weight 为连续 OIHW `[Cout,Cin,1,1]`，Bias 为 `[Cout]`。
模型整体 profile 保持 `[1,3,224,224]` 输入、eval、`[1,1000]` 输出。
本次验收范围是独立 pointwise 算子，接受离线 BN folding 后的 weight/bias。

## NCHW tile 寻址与真实 dot

M=H*W、N=Cout、K=Cin 仅定义逻辑矩阵维度，物理地址保持：

```text
A[m,k]   = X[k*(H*W)+m]
B[k,n]   = Weight[n*Cin+k]
Out[m,n] -> Out[n*(H*W)+m]
```

没有将 NCHW input reinterpret 成行连续 `[M,K]`，也没有把输出以 `[M,N]` 写回。
`program_id(0)` 选择 spatial tile，`program_id(1)` 选择 output-channel tile。
BM 在 M=1 时为 1，其余 case 为 16；BN=16，BK=32。
每次 K 迭代分别加载一个 `[BM,BK]` 的 NCHW 输入 tile 和 `[BK,BN]` 权重 tile，
用 `tl.dot(..., input_precision="ieee")` 累加至 FP32 accumulator，最后在同一个
Triton kernel 加 Bias 并按 NCHW 写回。

Input load 对 M/K 做 mask，Weight load 对 N/K 做 mask，Bias 对 N 做 mask，
Output store 对 M/N 做 mask。无效 dot lane 补零。
Cin=8/16/24 均小于 BK，Cin=40/72/88/120/144/240 等有 K 尾部；
H*W=196/49 有 M 尾部，Cout=8/24/40/72/88/120 等有 N 尾部。
比如 `pwconv_cin240_cout40_h14_w14` 的 M/N/K 尾部分别为 4/8/16。

编译器只物化固定大小的 tile 临时缓冲，最大单个 FP32 scratch 为 512 元素，
实际 allocation 形状限于 `[BM,BK]`、`[BK,BN]`、`[BM,BN]`、`[BN]`。
没有完整 im2col buffer 或独立 input/weight transpose kernel。

真实编译链：Triton ASTSource → TTIR 的 `tt.dot` → triton-riscv 的 `linalg.matmul`
→ Buddy bufferization → LLVM dialect → LLVM IR → RISC-V → NR ELF/BIN。
exporter 检查真实 dot/matmul 的存在，没有手写 `kernel.mlir`。

NR 复用 Qwen FP32 tile 使用的 Buddy
`--matmul-vectorization='vector-size=16 vector-type=fixed'`，生成实际 `vector.fma`，
再编译成 RVV FP32 FMA 指令。NR 向量化 LLVM 另在 Host 上执行完整 C oracle、
FP64/ATen 对照和逐 tile 检查，然后才编译、链接、审计 ELF。
这不是 AME 整数 matmul，也未宣称已在 FPGA 上运行。

## 全部 32 个 case 与 matcher 元数据

每个 case 的 `metadata.json` 都保存 `mnk: {M,N,K}`、NCHW/OIHW shape、
logical_addressing、constexprs、grid、tail 和误差界。
机器可读汇总见 [`inventory.json`](../validation/pointwise-conv2d/inventory.json)。

| case | M | N | K |
| --- | ---: | ---: | ---: |
| pwconv_cin16_cout8_h1_w1 | 1 | 8 | 16 |
| pwconv_cin8_cout16_h1_w1 | 1 | 16 | 8 |
| pwconv_cin16_cout16_h56_w56 | 3136 | 16 | 16 |
| pwconv_cin16_cout72_h56_w56 | 3136 | 72 | 16 |
| pwconv_cin72_cout24_h28_w28 | 784 | 24 | 72 |
| pwconv_cin24_cout88_h28_w28 | 784 | 88 | 24 |
| pwconv_cin88_cout24_h28_w28 | 784 | 24 | 88 |
| pwconv_cin24_cout96_h28_w28 | 784 | 96 | 24 |
| pwconv_cin96_cout24_h1_w1 | 1 | 24 | 96 |
| pwconv_cin24_cout96_h1_w1 | 1 | 96 | 24 |
| pwconv_cin96_cout40_h14_w14 | 196 | 40 | 96 |
| pwconv_cin40_cout240_h14_w14 | 196 | 240 | 40 |
| pwconv_cin240_cout64_h1_w1 | 1 | 64 | 240 |
| pwconv_cin64_cout240_h1_w1 | 1 | 240 | 64 |
| pwconv_cin240_cout40_h14_w14 | 196 | 40 | 240 |
| pwconv_cin40_cout120_h14_w14 | 196 | 120 | 40 |
| pwconv_cin120_cout32_h1_w1 | 1 | 32 | 120 |
| pwconv_cin32_cout120_h1_w1 | 1 | 120 | 32 |
| pwconv_cin120_cout48_h14_w14 | 196 | 48 | 120 |
| pwconv_cin48_cout144_h14_w14 | 196 | 144 | 48 |
| pwconv_cin144_cout40_h1_w1 | 1 | 40 | 144 |
| pwconv_cin40_cout144_h1_w1 | 1 | 144 | 40 |
| pwconv_cin144_cout48_h14_w14 | 196 | 48 | 144 |
| pwconv_cin48_cout288_h14_w14 | 196 | 288 | 48 |
| pwconv_cin288_cout72_h1_w1 | 1 | 72 | 288 |
| pwconv_cin72_cout288_h1_w1 | 1 | 288 | 72 |
| pwconv_cin288_cout96_h7_w7 | 49 | 96 | 288 |
| pwconv_cin96_cout576_h7_w7 | 49 | 576 | 96 |
| pwconv_cin576_cout144_h1_w1 | 1 | 144 | 576 |
| pwconv_cin144_cout576_h1_w1 | 1 | 576 | 144 |
| pwconv_cin576_cout96_h7_w7 | 49 | 96 | 576 |
| pwconv_cin576_cout1024_h1_w1 | 1 | 1024 | 576 |

32 组配置逐项匹配本地 `model.json` 中全部 41 次 pointwise convolution，
模块复用关系见 [`model-coverage.json`](../validation/pointwise-conv2d/model-coverage.json)。

## 独立 C oracle 和误差界

每个自动生成的 `launch.c` 使用独立的 `oc/h/w/ic` 四重 C loop，
直接按 NCHW Conv2D 公式以 double 累加，再加入 Bias[oc]，不调用矩阵乘法库。
9 轮覆盖 bias-only（含 ±0）、通道/空间 ramp、首/末输入通道、one-hot channel routing、
空间角点脉冲、强抵消、正负非二进制精确值和仅最后输出通道非零。

Host 四个输入/输出 buffer 的实际末尾各紧邻 PROT_NONE 页面；同时检查前缀 canary
及 X/Weight/Bias 逐位不变。NR 使用同一独立 C oracle，并检查前后 canary。

每个输出采用如下前向误差界，参数在运行前由 Cin 和 FP32 精度确定：

```text
u = 2^-24
gamma = ((Cin+2)*u)/(1-(Cin+2)*u)
A[oc,h,w] = sum_ic(abs(double(X[ic,h,w])*double(Weight[oc,ic]))) + abs(double(Bias[oc]))
bound[oc,h,w] = gamma*A[oc,h,w] + (Cin+2)*2^-149
abs(actual - double_reference) <= bound
```

每项最多经历一次 FP32 乘法、Cin-1 次加法、一次 bias 加法；额外一次 FP32
舍入余量覆盖 double reference 和界的求值误差。补齐 K lane 的零运算不增加误差，
不将容限按 padded K 放大。这个界也覆盖较少舍入的向量 FMA 累加。
分析范围为有限输入、round-to-nearest、无中间溢出；绝对项为下溢留余量。
使用绝对值乘积之和使抵消/接近零输出仍能正确验收，没有根据观测误差放宽阈值。

此外，每个 case 分别对普通 Host LLVM 和 NR 向量化 LLVM 执行 12 轮固定 seed=0
的 FP64 NCHW conv2d / `aten.convolution.default` 对照，包括 2^-4、1、2^4 的随机尺度。
两个 FP32 结果各自须满足 FP64 界，彼此差异不得超过两倍界。
首/中/末 spatial tile 和 output-channel tile 的组合还单独调用真实编译入口，
检查只写该 tile 对应的 NCHW 区间。每种 LLVM 共通过 180 次独立 tile 检查。

| 实际验证 | 结果 |
| --- | --- |
| 独立 C oracle，32 case × 9 轮 | 32/32 PASS |
| 普通 Host LLVM 的 FP64/ATen 对照 | 32/32 PASS；8,257,248 个输出样本 |
| NR 向量化 LLVM 的 C + FP64/ATen 对照 | 32/32 PASS |
| 两种 LLVM 的独立 tile 写入检查 | 180 + 180 PASS |
| C 最大绝对误差 / 最大 error-bound 比值 | 5.44637441635e-5 / 0.130120394 |
| FP64/ATen 压力测试最大绝对误差 / 最大 error-bound 比值 | 0.00467350145 / 0.257209937 |
| 原有 55 个 case Host 回归 | 55/55 PASS |
| NR ELF/BIN 构建与指令审计 | 32/32 PASS |
| FPGA 实际运行 | **NOT EXECUTED** |

逐 case 误差、日志及 hash 见 `validation/pointwise-conv2d/host.json` 和 `nr.json`。
压力测试放大了输入和权重，不能将其绝对误差直接与 C 小尺度样本比较。
Host 与 NR 向量 LLVM 在本机的误差统计相同，也不等于 FPGA 上已通过。

## 复现命令和产物

仓库根目录使用本次实际验证的本机工具：

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
P=/home/zhangwenji/triton-riscv/.venv/bin/python
M=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k

# 从 inventory 自动生成全部 32 个目录，每个包含 metadata.json / launch.c / makefile。
"$P" "$M/triton/cases.py" --family=pointwise_conv2d --generate

# 批量执行全部 32 个 Host 验证。
"$P" "$M/triton/build.py" --family=pointwise_conv2d --host

# 仅当全部 32 个 Host PASS 后才构建 NR；内部另验证 NR 向量化 LLVM。
"$P" "$M/triton/build.py" --family=pointwise_conv2d --nr
```

make 入口也支持 `make -C "$M/triton" FAMILY=pointwise_conv2d TRITON_PYTHON="$P" check`
和 `all`。工具来自本机已安装的外部 build；实际路径、版本、参数和 SHA-256 记录在
`validation/pointwise-conv2d/toolchain.json`，未声称重建了本仓库 pinned toolchain。

全部 32 个 case 的 ELF/BIN 已生成，路径对应上表 case 名：

```text
$M/triton/build/<case>/nr/<case>.elf
$M/triton/build/<case>/nr/<case>.bin
```

以 `CASE=pwconv_cin40_cout240_h14_w14`、`B=$M/triton/build/$CASE` 为例：

| 产物 | 路径 |
| --- | --- |
| TTIR | `$B/kernel.ttir` |
| triton-riscv Linalg MLIR | `$B/kernel.linalg.mlir` |
| NR 向量化 MLIR | `$B/nr/vectorized.mlir` |
| NR LLVM dialect / LLVM IR | `$B/nr/kernel.llvm.mlir` / `$B/nr/kernel.ll` |
| NR assembly | `$B/nr/kernel.nr.S` |
| ELF，32,080 bytes | `$B/nr/pwconv_cin40_cout240_h14_w14.elf` |
| BIN，77,507 bytes | `$B/nr/pwconv_cin40_cout240_h14_w14.bin` |
| Host C / NR-vector Host C 日志 | `$B/host/output.log` / `$B/nr/vector-host.log` |
| ELF 审计 | `$B/nr/elf-audit.json` |

按此前暂停上板、保留本地产物的要求，本次未执行 SSH/scp/FPGA 操作。
未来上板复用原脚本，例如下面的命令；本次未执行：

```bash
CASE=pwconv_cin40_cout240_h14_w14
examples/FPGA-BOSCAME/fpga_run.sh \
  "$M/triton/build/$CASE/nr/$CASE.bin" \
  --fpga=5 --capture-seconds=120 --completion-marker='[nr] RA returned:'
```

全部上板可用 `"$P" "$M/triton/build.py" --family=pointwise_conv2d --run`，
仍复用同一个脚本，任何 case 未通过就停止。只有实际捕获数值 PASS、
`[nr] RA returned: PASS` 且脚本成功退出，才记录 FPGA PASS。

## 本次文件

修改：`triton/kernels.py`、`triton/cases.py`、`triton/export.py`、`triton/build.py`、
`common.mk`、根 `README.md` 和 `triton/README.md`。
新增：`triton/launch_pointwise_conv2d.c.in`、`triton/verify_pointwise_conv2d.py`、
本文档，32 个 case 的 `launch.c / metadata.json / makefile`，以及
`validation/pointwise-conv2d/` 下的 host/nr/regression/model-coverage/inventory/toolchain/status 记录。
Qwen、公共 runtime、`support.c/h` 和 FPGA 脚本均未修改。

实现、全部 Host 验证和 NR binary 构建：**COMPLETE**。
全部 32 个 case 的 FPGA 实际结果：**NOT EXECUTED**，尚无板上数值通过的结论。
