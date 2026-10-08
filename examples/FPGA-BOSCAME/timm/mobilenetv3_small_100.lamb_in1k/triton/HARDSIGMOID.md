# Hardsigmoid：7 个静态 Triton cases

> 2026-10-06 后续上板结果见 [逐 case 验证记录](../validation/fpga-20261006/README.md)。下文的 `NOT EXECUTED` 为初次构建时的历史状态；patch5 与 patch6 的结果分别记录。

使用此前合并后的目录 `examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/`。
本次只新增 `kernels.py::hardsigmoid` 这一类算子，统一处理连续 NCHW FP32 张量。
固定模型 profile 保持为 batch=1、输入 `[1,3,224,224]`、eval、输出 `[1,1000]`。

## 实现与 case 清单

语义为 `clamp(x + 3, 0, 6) / 6`，对应
[PyTorch 2.10 的 CPU 实现](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/Activation.cpp#L504)。
Triton 显式保留 NaN 传播，使用 `tl.div_rn(clamped, 6.0)` 保留 FP32 除法舍入，
不将除法替换为乘以预先舍入的 `1/6`。

只有一个通用 kernel，COUNT 和 BLOCK 为 constexpr；BLOCK=128，
grid 为 `[ceil(COUNT/128),1,1]`，load/store 均使用 `index < COUNT` mask。
本类无卷积权重、OIHW 或空间 padding；展平保持原 NCHW 元素顺序，
无效 tile 元素只在寄存器/临时张量中补零，不读取或写入原张量边界之外。

| case | NCHW | COUNT | grid.x | 尾部有效元素 |
| --- | --- | ---: | ---: | ---: |
| hardsigmoid_count16 | [1,16,1,1] | 16 | 1 | 16 |
| hardsigmoid_count96 | [1,96,1,1] | 96 | 1 | 96 |
| hardsigmoid_count240 | [1,240,1,1] | 240 | 2 | 112 |
| hardsigmoid_count120 | [1,120,1,1] | 120 | 1 | 120 |
| hardsigmoid_count144 | [1,144,1,1] | 144 | 2 | 16 |
| hardsigmoid_count288 | [1,288,1,1] | 288 | 3 | 32 |
| hardsigmoid_count576 | [1,576,1,1] | 576 | 5 | 64 |

清单与 `model.json` 中 9 次 `aten.hardsigmoid.default` 调用的 7 个去重
COUNT 一致。`cases.py --family=hardsigmoid --generate` 从统一 inventory
及 `launch_hardsigmoid.c.in` 生成全部目录，每个包含 `launch.c`、
`metadata.json`、`makefile`。正常 build 也自动执行生成。

## 验证

每个 `launch.c` 在调用实际编译产物前，以独立 C 循环计算 reference。
三轮输入的头部、尾部均覆盖以下五个区域，并显式检查每个区域确实出现：
`x < -3`、`x == -3`、`-3 < x < 3`、`x == 3`、`x > 3`。
同时包含 ±3 的相邻可表示 FP32 值、正负零、±FLT_MAX、±Infinity 和 NaN；
最小 COUNT=16 也完整覆盖。有限输出采用精确数值比较，NaN 比较类别，
不要求 NaN payload 一致。输入不可变性按位检查。

Host 缓冲区紧邻 PROT_NONE 页，可发现 masked load/store 的尾部越界；
前方保留 canary，NR 缓冲区前后各有 128 个 guard。所有 case 都包含非整 tile。

`verify_hardsigmoid.py` 另外将相同的 Host LLVM IR 与 ABI adapter 链接成动态库，
通过 ctypes 调用真实编译产物，并与 `torch.ops.aten.hardsigmoid.default` 比较。
每个 case 使用 seed=0 的 35 轮输入：3 轮边界值、16 轮 `[-8,8)` 均匀随机值、
16 轮随机 FP32 位模式。输入采用对应 NCHW shape，CPU FP32 inference mode。
该检查补充独立 C oracle，不替换它；任一不一致都会令 Host 构建失败。

实际结果：

- 独立 C Host 检查：**7/7 PASS**，最大绝对误差全为 **0**。
- PyTorch 2.10.0+cpu 对照：**7/7 PASS**，共 **51,800** 个样本，最大绝对误差 **0**。
- NR ELF/BIN 构建及 ELF 指令审计：**7/7 PASS**。
- 共享脚本回归：residual_add **4/4 Host PASS**，ReLU **11/11 Host PASS**。
- FPGA：**NOT EXECUTED**，沿用用户暂不连接/上板的要求。
- 状态：**COMPLETE**，范围为本类实现、Host 验证和 NR binary 构建。

证据：[host.json](../validation/hardsigmoid/host.json)、
[nr.json](../validation/hardsigmoid/nr.json)、
[toolchain.json](../validation/hardsigmoid/toolchain.json)、
[status.json](../validation/hardsigmoid/status.json)。

## 构建命令

仓库根目录执行本机已验证的命令：

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
MOBILENET_PYTHON=/home/zhangwenji/triton-riscv/.venv/bin/python
MOBILENET_TRITON=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton

"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=hardsigmoid --host
"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=hardsigmoid --nr
```

`--nr` 首先验收全部 7 个 Host case；只有源码、工具和产物摘要匹配时才复用
已有 PASS 证据，之后才编译 NR runtime 和所有二进制。任一失败立即停止。
也支持以下 Make 入口；默认 family 仍为 residual_add：

```bash
make -C "$MOBILENET_TRITON" FAMILY=hardsigmoid check TRITON_PYTHON="$MOBILENET_PYTHON"
make -C "$MOBILENET_TRITON" FAMILY=hardsigmoid all TRITON_PYTHON="$MOBILENET_PYTHON"
```

实际链路：Triton AST → TTIR → triton-riscv `triton-to-linalg-experimental`
→ Linalg → Buddy bufferization/lowering → LLVM → RISC-V → NR ELF/BIN。
Linalg 中实际存在 `arith.addf`、`arith.maximumf`、`arith.minimumf`、`arith.divf`，
NR 汇编含 `fadd.s`、`fmax.s`、`fmin.s`、`fdiv.s` 及 NaN 传播处理。
没有手写 `kernel.mlir`，没有修改 Qwen3、公共 runtime 或上板脚本。
使用本机已安装的外部 Triton/Buddy/LLVM 工具，版本及 SHA-256 均记录在证据中，
不声称匹配 Qwen3 的锁定工具链版本。

## 产物与上板入口

每个 case 都有 `triton/build/<case>/` 下的实际产物：

| 产物 | 相对路径 |
| --- | --- |
| TTIR | `kernel.ttir` |
| Linalg MLIR | `kernel.linalg.mlir` |
| 前端 provenance / ABI adapter | `frontend.json`、`adapter.c` |
| Host LLVM IR / C oracle / 日志 | `host/kernel.ll`、`host/check`、`host/output.log` |
| PyTorch 实际编译产物对照 | `host/pytorch-check.so`、`host/pytorch.json` |
| NR LLVM dialect / LLVM IR | `nr/kernel.llvm.mlir`、`nr/kernel.ll` |
| RISC-V 汇编 | `nr/kernel.nr.S` |
| ELF / BIN | `nr/<case>.elf`、`nr/<case>.bin` |
| ELF 审计 | `nr/elf-audit.json` |

例如非整 tile case `hardsigmoid_count144` 的 BIN：
`triton/build/hardsigmoid_count144/nr/hardsigmoid_count144.bin`。
以下命令供后续使用，**本次未执行**：

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton/build/hardsigmoid_count144/nr/hardsigmoid_count144.bin \
  --fpga=5 --capture-seconds=120 --completion-marker='[nr] RA returned:'
```

`build.py --family=hardsigmoid --run` 会依次复用同一脚本，要求脚本成功退出、
各 case 数值 PASS 和 `[nr] RA returned: PASS` 同时出现才记录 FPGA PASS。
本次没有执行此入口，也没有创建 FPGA PASS 记录。
