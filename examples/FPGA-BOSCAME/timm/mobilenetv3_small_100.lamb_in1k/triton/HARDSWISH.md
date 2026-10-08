# Hardswish：9 个静态 Triton cases

> 2026-10-06 后续上板结果见 [逐 case 验证记录](../validation/fpga-20261006/README.md)。下文的 `NOT EXECUTED` 为初次构建时的历史状态；patch5 与 patch6 的结果分别记录。

实现位于此前合并后的 `examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/`。
本次仅新增一个通用 Triton kernel：`kernels.py::hardswish`。
模型 profile 保持 batch=1、输入 `[1,3,224,224]`、NCHW、FP32、eval、输出 `[1,1000]`。

## 融合与数值语义

每个元素在同一个 Triton kernel 中完成
`x * clamp(x + 3, 0, 6) / 6`，只有一次输入 load 和一次输出 store，
没有将 add、clamp、mul、div 拆成多个 kernel，也没有调用 hardsigmoid kernel。
保留 [PyTorch 2.10 CPU 实现](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/Activation.cpp#L708)
的 FP32 运算顺序：加法、clamp、乘法、除法；使用 `tl.div_rn`，关闭浮点融合。
因此保留乘法中间结果的舍入及溢出行为，不重排为 `x * (clamp(...) / 6)`。
NaN 显式传播；`-Infinity * 0` 为 NaN，`+FLT_MAX * 6` 溢出为正 Infinity。

COUNT、BLOCK 为 constexpr；BLOCK=128，输入/输出均为连续 NCHW FP32，
统一展平后按 COUNT specialization，grid 为 `[ceil(COUNT/128),1,1]`。
load/store 都有 `index < COUNT` mask。此类算子无卷积权重、OIHW 或空间 padding；
无效 tile lane 的补零不读写原张量边界以外的内存。

## 全部 cases

| case | 来源 NCHW | grid.x | 最后一个 tile 有效元素 |
| --- | --- | ---: | ---: |
| hardswish_count200704 | [1,16,112,112] | 1568 | 128 |
| hardswish_count75264 | [1,96,28,28] | 588 | 128 |
| hardswish_count18816 | [1,96,14,14] | 147 | 128 |
| hardswish_count47040 | [1,240,14,14] | 368 | 64 |
| hardswish_count23520 | [1,120,14,14] | 184 | 96 |
| hardswish_count28224 | [1,144,14,14]、[1,576,7,7] | 221 | 64 |
| hardswish_count56448 | [1,288,14,14] | 441 | 128 |
| hardswish_count14112 | [1,288,7,7] | 111 | 32 |
| hardswish_count1024 | [1,1024,1,1] | 8 | 128 |

原始 `model.json` 中 19 次 `aten.hardswish_.default` 调用有 10 个不同 NCHW
shape，去重后为上述 9 个 COUNT。这里实现输出到独立 Out 的相同逐元素数值语义。
`COUNT=28224` 的两种 shape 共用同一份 TTIR、Linalg、LLVM、ELF/BIN。
`cases.py --family=hardswish --generate` 从唯一 inventory 和 `launch_hardswish.c.in`
自动生成全部 9 个目录，各含 `launch.c`、`metadata.json`、`makefile`；build 也自动生成。

## Host 验证

每个 `launch.c` 在调用编译产物前，用独立 C 循环计算 reference，严格先乘再除。
三轮输入的头部和尾部都包括 ±3 及两侧相邻 FP32 值、正负零、正负极值、
NaN 和 Infinity；其余元素为覆盖 `[-8,8]` 的确定性序列。
每轮显式断言五个区域均存在：`x < -3`、`x == -3`、`-3 < x < 3`、`x == 3`、`x > 3`。
有限值使用精确数值比较，允许正负零相等；NaN 比较类别，Infinity 比较符号。
输入不可变性按位检查。没有使用 Triton 输出作为 reference。

Host 缓冲区在 COUNT 后紧邻 PROT_NONE 页，检查尾部 load/store 越界；前方保留
canary。NR 缓冲区前后各保留 128 个 guard。9 个 case 同时覆盖整 tile 和非整 tile。

`verify_hardswish.py` 将同一个 Host LLVM IR 与 ABI adapter 链接成动态库，
通过 ctypes 调用实际编译产物，并与 `torch.ops.aten.hardswish.default` 精确比较。
每个来源 shape 有 35 轮测试：3 轮边界模式、16 轮 `[-8,8)` 随机值、
16 轮随机 FP32 位模式；seed=0、CPU FP32 inference mode。
COUNT=28224 的两个 shape 分别测试，使用同一个动态库入口。
比较包括 NaN/Infinity 分类和输入不可变性，不对无穷相减计算绝对误差。
这一检查补充独立 C oracle，任一检查失败均停止。

实际结果：

- 独立 C Host 验证：**9/9 PASS**，有限输出的最大绝对误差均为 **0**。
- PyTorch 2.10.0+cpu 对照：**9 个 case、10 个 shape 全部 PASS**，
  **17,268,160** 个样本，有限输出最大绝对误差 **0**，非有限值类别/符号一致。
- NR ELF/BIN 构建及 ELF 指令审计：**9/9 PASS**。
- 共享脚本 Host 回归：residual_add **4/4**、ReLU **11/11**、hardsigmoid **7/7 PASS**。
  回归产物在 `triton/build/regression-hardswish/`，保留此前各类的产物和验证报告。
- FPGA：**NOT EXECUTED**，沿用用户暂停上板的约定。
- 本次状态：**COMPLETE**，范围为 hardswish 实现、Host 验证和 NR 二进制构建。

证据：[host.json](../validation/hardswish/host.json)、[nr.json](../validation/hardswish/nr.json)、
[regression.json](../validation/hardswish/regression.json)、
[status.json](../validation/hardswish/status.json)。

## 构建命令

在仓库根目录执行本机已验证的命令：

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
MOBILENET_PYTHON=/home/zhangwenji/triton-riscv/.venv/bin/python
MOBILENET_TRITON=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton

"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=hardswish --host
"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=hardswish --nr
```

`--nr` 先完成全部 9 个 Host 验收，再构建 NR runtime 和二进制；仅在源码、
工具及产物摘要匹配时复用已有 PASS 证据。任一失败立即停止。
也支持以下 Make 入口，默认 family 仍为 residual_add：

```bash
make -C "$MOBILENET_TRITON" FAMILY=hardswish check TRITON_PYTHON="$MOBILENET_PYTHON"
make -C "$MOBILENET_TRITON" FAMILY=hardswish all TRITON_PYTHON="$MOBILENET_PYTHON"
```

实际链路为 Triton AST → TTIR → triton-riscv `triton-to-linalg-experimental`
→ Linalg → Buddy bufferization/lowering → LLVM → RISC-V → NR ELF/BIN。
每份 TTIR 只有一个 hardswish 入口；Linalg 中的 add、max、min、mul、div
仍位于同一个函数内，ABI adapter 仅负责描述符和 grid 调用，不做张量数值计算。
NR 汇编包含 `fadd.s`、`fmax.s`、`fmin.s`、`fmul.s`、`fdiv.s` 及 NaN 传播处理。
没有手写 `kernel.mlir`，Qwen3、公共 runtime 和上板脚本均未修改。
本机使用已安装的外部 Triton/Buddy/LLVM；实际路径、版本及 SHA-256 见
[toolchain.json](../validation/hardswish/toolchain.json)，不声称匹配 Qwen3 的工具链锁定版本。

## 产物和上板入口

每个 case 的实际产物在 `triton/build/<case>/` 下：

| 产物 | 相对路径 |
| --- | --- |
| TTIR | `kernel.ttir` |
| Linalg MLIR | `kernel.linalg.mlir` |
| 前端记录 / ABI adapter | `frontend.json`、`adapter.c` |
| Host LLVM / C oracle 可执行文件 / 日志 | `host/kernel.ll`、`host/check`、`host/output.log` |
| PyTorch 对照库 / 报告 | `host/pytorch-check.so`、`host/pytorch.json` |
| NR LLVM dialect / LLVM IR | `nr/kernel.llvm.mlir`、`nr/kernel.ll` |
| RISC-V 汇编 | `nr/kernel.nr.S` |
| ELF / BIN | `nr/<case>.elf`、`nr/<case>.bin` |
| ELF 审计 | `nr/elf-audit.json` |

非整 tile 示例 `hardswish_count23520` 的 BIN 为
`triton/build/hardswish_count23520/nr/hardswish_count23520.bin`。
以下命令供后续使用，**本次未执行**：

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton/build/hardswish_count23520/nr/hardswish_count23520.bin \
  --fpga=5 --capture-seconds=120 --completion-marker='[nr] RA returned:'
```

`build.py --family=hardswish --run` 可依次复用同一个公共脚本。
只有脚本成功退出，同时出现该 case 的数值 PASS 和 `[nr] RA returned: PASS`，
才记录 FPGA PASS。本次没有连接服务器或执行上板程序。
