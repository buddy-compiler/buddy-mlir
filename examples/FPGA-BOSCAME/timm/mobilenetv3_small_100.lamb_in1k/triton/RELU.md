# ReLU：11 个静态 Triton cases

> 2026-10-06 后续上板结果见 [逐 case 验证记录](../validation/fpga-20261006/README.md)。下文的 `NOT EXECUTED` 为初次构建时的历史状态；patch5 与 patch6 的结果分别记录。

实现位于之前合并后的 `examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/`。
唯一新增的 kernel 为 `kernels.py::relu`，语义为 FP32
`Out[i] = max(X[i], 0)`。输入/输出是相同 shape 的连续 NCHW 张量，
统一展平为 COUNT，BLOCK 固定 specialization 为 128，load/store 均带 mask。
没有手写 `kernel.mlir`，没有新增其他类别的算子。

## 全部 cases

| case | NCHW | COUNT | grid.x | 最后一块有效元素 |
| --- | --- | ---: | ---: | ---: |
| relu_count50176 | [1,16,56,56] | 50176 | 392 | 128 |
| relu_count8 | [1,8,1,1] | 8 | 1 | 8 |
| relu_count225792 | [1,72,56,56] | 225792 | 1764 | 128 |
| relu_count56448 | [1,72,28,28] | 56448 | 441 | 128 |
| relu_count68992 | [1,88,28,28] | 68992 | 539 | 128 |
| relu_count24 | [1,24,1,1] | 24 | 1 | 24 |
| relu_count64 | [1,64,1,1] | 64 | 1 | 64 |
| relu_count32 | [1,32,1,1] | 32 | 1 | 32 |
| relu_count40 | [1,40,1,1] | 40 | 1 | 40 |
| relu_count72 | [1,72,1,1] | 72 | 1 | 72 |
| relu_count144 | [1,144,1,1] | 144 | 2 | 16 |

完整 grid 为 `[grid.x,1,1]`。清单与原始模型 `model.json` 中全部 ReLU 的
去重 COUNT 相符。`cases.py --family=relu --generate` 从统一 inventory
和 `launch_relu.c.in` 生成全部 11 个目录；每个目录包含独立的
`launch.c`、`metadata.json`、`makefile`。

## 数值和边界验收

每个 `launch.c` 用独立 C 循环，在调用编译产物之前计算
`reference[i] = X[i] > 0.0f ? X[i] : 0.0f`，不是使用 Triton 结果作为 reference。
两轮不同输入均含负数、`+0.0`、`-0.0`、正数及 `±FLT_MAX`。
边界值在开头和尾部重复出现；即使 COUNT=8 也完整覆盖。
C 测试显式检查正数、负数、两种零确实出现，防止测试数据遗漏。

输出按 FP32 数值精确比较，因此 `+0.0 == -0.0` 不会误报。
输入按位检查不可被修改，能发现输入的 `-0.0` 被意外改成 `+0.0`。
验收范围是上述有限 FP32 输入；本次未验收 NaN/Infinity 行为。

Host 将 `X[COUNT]` 和 `Out[COUNT]` 起始页设为不可访问，检查 masked
load/store 的尾部越界；前部保留 guard。NR 在缓冲区前后各设 128 个 guard。
ReLU 无卷积权重、OIHW 或空间 padding；COUNT mask 覆盖 tile 的尾部补位。

## 构建和验证

复用 [现有构建链](README.md)：Triton AST → TTIR → triton-riscv 的
`triton-to-linalg-experimental` → Linalg → Buddy → LLVM → RISC-V。
实际 ReLU Linalg 内含 `linalg.generic` 和 `arith.maxnumf`，NR 生成 `fmax.s`。
共用 ABI 适配和 ABI 审计已支持 ReLU 的两个 tensor 参数，仍支持 residual add
的三个参数；已有 residual add 的四项 Host 回归通过。

在仓库根目录运行本机已经验证的命令：

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
MOBILENET_PYTHON=/home/zhangwenji/triton-riscv/.venv/bin/python
MOBILENET_TRITON=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton

"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=relu --host
"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=relu --nr
```

`--nr` 在所有 11 项 Host 验收通过后才开始 NR 编译；旧验收结果只有在源码、
工具、IR、可执行文件和日志摘要匹配时才复用。任意失败均停止。
工具实际路径、版本、SHA-256 见 [`toolchain.json`](../validation/relu/toolchain.json)。
本机使用已安装的外部 Triton/Buddy/LLVM 构建，不声称匹配 Qwen3 的工具链锁定版本。

Make 入口为 `make -C "$MOBILENET_TRITON" check FAMILY=relu TRITON_PYTHON=...`
及 `all FAMILY=relu`；从任一 `relu_count*/` 目录执行 `make check/all`
同样会验收/构建全部 ReLU cases。默认 family 保持为原有 residual_add。

## 实际结果

- **Host：11/11 PASS**，所有输出最大绝对误差均为 0。
- **NR build：11/11 PASS**，全部 ELF 通过仓库公共指令审计。
- **residual_add Host 回归：4/4 PASS**。
- **FPGA：11/11 NOT EXECUTED**，按用户要求本次没有进行上板验证。
- **本次状态：COMPLETE**，范围为实现、Host 验收和 NR 构建；不代表 FPGA PASS。

证据：[host.json](../validation/relu/host.json)、[nr.json](../validation/relu/nr.json)、
[status.json](../validation/relu/status.json)。Qwen3、公共 runtime 和上板脚本未修改。

## 产物与后续上板

每个 case 的实际产物目录为 `triton/build/<case>/`：

| 产物 | 相对路径 |
| --- | --- |
| TTIR | `kernel.ttir` |
| Linalg MLIR | `kernel.linalg.mlir` |
| 前端记录与 ABI 适配 | `frontend.json`、`adapter.c` |
| Host LLVM / 可执行文件 / 日志 | `host/kernel.ll`、`host/check`、`host/output.log` |
| NR LLVM dialect / LLVM IR | `nr/kernel.llvm.mlir`、`nr/kernel.ll` |
| ELF / BIN | `nr/<case>.elf`、`nr/<case>.bin` |
| ELF 审计 | `nr/elf-audit.json` |

例如 `relu_count8` 的 BIN 为 `triton/build/relu_count8/nr/relu_count8.bin`。
以下为后续可用的上板命令，本次未执行：

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton/build/relu_count8/nr/relu_count8.bin \
  --fpga=5 --capture-seconds=120 --completion-marker='[nr] RA returned:'
```

`build.py --family=relu --run` 可在后续获准上板时依次调用同一个公共脚本。
它要求 runner 成功退出，并实际看到各 case 数值 PASS 和 RA 返回 PASS。
