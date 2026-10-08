# Residual add：Triton → NR FPGA

> 2026-10-06 后续上板结果见 [逐 case 验证记录](../validation/fpga-20261006/README.md)。下文的 `NOT EXECUTED` 为初次构建时的历史状态；patch5 与 patch6 的结果分别记录。

Stem 3×3 dense Conv2d 使用 `--family=conv_stem`，唯一 case 的 NCHW/OIHW、
padding、K=27 mask、独立 C oracle 和 NR 产物见 [STEM.md](STEM.md)。

NCHW pointwise Conv2d 使用 `--family=pointwise_conv2d`，全部 32 个 case 的
M/N/K、NCHW dot 寻址、Host 验证和 NR 产物见 [PWCONV.md](PWCONV.md)。

NCHW depthwise Conv2d 使用 `--family=depthwise_conv2d`，9 个 case 的
padding/stride/channel 布局、独立 C oracle 和 NR 产物见 [DWCONV.md](DWCONV.md)。

Classifier Linear 使用 `--family=linear`，唯一 case `linear_m1_n1000_k1024`，
直接读取 `[N,K]`、融合 bias 的 FP32 dot 与 NR RVV 验证见 [LINEAR.md](LINEAR.md)。

H/W mean reduction 使用 `--family=mean_hw`，7 个 case、独立 double oracle 和
误差容限依据见 [MEAN_HW.md](MEAN_HW.md)。

SE channel broadcast multiply 使用 `--family=se_mul`，7 个 case 的清单、
NCHW/Scale 布局和验证见 [SE_MUL.md](SE_MUL.md)。

Hardswish 使用 `--family=hardswish`，其 9 个 COUNT case、验收和产物见
[HARDSWISH.md](HARDSWISH.md)。

Hardsigmoid 使用 `--family=hardsigmoid`，其 7 个 case、验收和产物见
[HARDSIGMOID.md](HARDSIGMOID.md)。

ReLU 使用同一构建入口的 `--family=relu`，其 11 个 case、验收和产物见
[RELU.md](RELU.md)。本页描述默认的 `residual_add` family。

只实现一个 `kernels.py::residual_add`：`Out[i] = X[i] + Y[i]`。
输入和输出均为连续 NCHW FP32，展平后按 COUNT specialization。
BLOCK 是 constexpr，当前为 128；两次 load 和一次 store 都使用 `i < COUNT` mask。
这类算子没有卷积权重或空间 padding，NCHW 展平不会改变元素顺序。

## 全部静态 cases

| case | NCHW | COUNT | grid | 最后一个 block 有效元素 |
| --- | --- | ---: | --- | ---: |
| add_count18816 | [1,24,28,28] | 18816 | [147,1,1] | 128 |
| add_count7840 | [1,40,14,14] | 7840 | [62,1,1] | 32 |
| add_count9408 | [1,48,14,14] | 9408 | [74,1,1] | 64 |
| add_count4704 | [1,96,7,7] | 4704 | [37,1,1] | 96 |

`cases.py` 是唯一 inventory，按 COUNT 去重；同元素总数的 NCHW shape
会追加到同一个 case 的 shapes 列表。它从 `launch.c.in` 自动生成上级四个
case 目录内的 `launch.c`、`metadata.json`、`makefile`，没有手写 `kernel.mlir`。
运行 `python cases.py --generate` 可重新生成，正常 build 会自动执行生成。

## 实际编译链

```text
kernels.py (@triton.jit)
  → ASTSource.make_ir → CPUBackend.make_ttir
  → kernel.ttir
  → triton-shared-opt --triton-to-linalg-experimental
  → kernel.linalg.mlir（含真实 linalg.generic + arith.addf）
  → Buddy bufferization / stack promotion / LLVM lowering
  → kernel.llvm.mlir → buddy-translate → kernel.ll
  → llc RISC-V scalar FP32 → kernel.s
  → 仓库现有 assembly 编码/约束工具 → kernel.nr.S
  → clang + common/nr runtime + nr.ld → ELF
  → check_nr_elf.py → llvm-objcopy → BIN
  → 仓库原有 fpga_run.sh --fpga=5
```

`export.py` 只做真正的 Triton 编译、符号命名和 memref/grid ABI 包装；
不生成算术 MLIR。显式调用 triton-riscv 的标准 Linalg pipeline，避免本机
backend 默认选择已完成向量化的 CPU 输出而跳过可审阅的 Linalg 产物。
Host 和 NR 从同一份 `kernel.linalg.mlir` 降低。

本次 NR 使用 scalar FP32；residual add 不含矩阵运算，也不需要 AME 指令。
编译器生成的 masked subview 复制通过标准 `memrefCopy` ABI 执行：
NR 复用 `common/nr/nr_runtime.c`，Host 支持代码提供 rank-1 复制适配。
这些复制只搬运数据，求和计算来自 Triton。中间缓冲区提升到栈，禁止每个
program 调用 malloc/aligned_alloc/free。

`common.mk` 直接包含共享 `common/toolchain.mk` 和 `common/nr/nr.mk`。
启动、链接、串口、上传、上板脚本均使用已有实现，Qwen3 文件保持原样。

## Host oracle 和边界检查

每个生成的 `launch.c` 使用独立 C 循环提前计算 `reference[i]=X[i]+Y[i]`。
执行两组不同的正负、非零输入，并检查每个输出元素、输入未被改写和 guard。
测试数值为二进制可精确表示的小数，因此验收要求 FP32 精确相等，零误差。

Host 在三个缓冲区的 `COUNT` 边界后设置 `PROT_NONE` 页：未正确 mask 的
尾部 load/store 会直接失败，前部另有 128 个 guard 值。
NR 在三个缓冲区的前后各放置 128 个 guard，全部检查。
所有 four cases Host PASS 是 NR 编译的前置条件；源码、工具或 IR 摘要变化
会使旧 Host 证据失效。没有通过时脚本退出，不继续 NR 构建。

## 本机已执行的命令

在仓库根目录运行。这台机器使用既有外部 Triton/Buddy/LLVM 构建，
没有假定当前 checkout 已有编译工具；工具实际路径、版本和 SHA-256
完整记录在 [`toolchain.json`](../validation/residual-add/toolchain.json)。
这些是本次实际使用的工具，不声称匹配 Qwen3 的 toolchain-lock。

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
MOBILENET_PYTHON=/home/zhangwenji/triton-riscv/.venv/bin/python
MOBILENET_TRITON=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton

# 四项 Host 检查。
"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --host

# 先检查四项 Host 证据，再构建全部 FPGA ELF/BIN。
"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --nr

# 认证可用后：校验/构建并依次调用已有 fpga_run.sh。
"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --run
```

也可用 `make -C "$MOBILENET_TRITON" check/all/run TRITON_PYTHON=...`。
在其他已构建环境中覆盖工具路径即可；`TRITON_SHARED_OPT_PATH` 必须指向
真实 triton-riscv converter。`--run` 遇到第一个未通过的 case 就停止，
未运行的 case 明确保留 `NOT EXECUTED`。

手工单项上板命令如下；四个 case 仅替换 case 名称：

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton/build/add_count18816/nr/add_count18816.bin \
  --fpga=5 --capture-seconds=120 --completion-marker='[nr] RA returned:'
```

脚本要求 runner 成功退出，并同时看到 `verify <case>: PASS errors=00000000`
和 `[nr] RA returned: PASS`，才记录 FPGA PASS。

## 实际结果与产物

| case | Host | max_abs_error | NR build / ELF 审计 | FPGA |
| --- | --- | ---: | --- | --- |
| add_count18816 | PASS | 0 | PASS | NOT EXECUTED |
| add_count7840 | PASS | 0 | PASS | NOT EXECUTED |
| add_count9408 | PASS | 0 | PASS | NOT EXECUTED |
| add_count4704 | PASS | 0 | PASS | NOT EXECUTED |

本次 `fpga_run.sh` 尝试第一个镜像时，现有 `fpga` SSH 别名在上传前返回
`Permission denied (publickey,password)`，重试耗尽；其他三个 case 未上板。
没有 FPGA PASS。当前整项任务状态为 **INCOMPLETE**，待真实板上验收。

证据：[`host.json`](../validation/residual-add/host.json)、
[`nr.json`](../validation/residual-add/nr.json)、
[`fpga.json`](../validation/residual-add/fpga.json)、
[`上板失败日志`](../validation/residual-add/add_count18816.fpga.log)。
每个前端 manifest 记录 Triton 源码、backend、converter 和 IR 摘要；
Host/NR 清单记录 oracle、公共运行时、构建配置及实际二进制摘要。

产物统一位于 `triton/build/<case>/`，被已有 `.gitignore` 忽略：

目录已合并到 `examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/`。
历史 FPGA 认证失败日志及其命令保留原样，其中的旧路径对应迁移前的位置；
迁移不会改变 `NOT EXECUTED` 状态，重新上板请使用上面的当前路径。

| 产物 | 相对于 `triton/build/<case>/` 的路径 |
| --- | --- |
| TTIR | `kernel.ttir` |
| Linalg MLIR | `kernel.linalg.mlir` |
| 前端来源记录 / ABI 适配 | `frontend.json` / `adapter.c` |
| Host LLVM IR / 可执行文件 / 日志 | `host/kernel.ll` / `host/check` / `host/output.log` |
| NR LLVM dialect / LLVM IR | `nr/kernel.llvm.mlir` / `nr/kernel.ll` |
| NR 汇编 / 指令约束后汇编 | `nr/kernel.s` / `nr/kernel.nr.S` |
| ELF / BIN | `nr/<case>.elf` / `nr/<case>.bin` |
| ELF 审计 | `nr/elf-audit.json` |

例如 `add_count18816` 的最终 BIN 为
`triton/build/add_count18816/nr/add_count18816.bin`；其余三项路径遵循同一规则。
