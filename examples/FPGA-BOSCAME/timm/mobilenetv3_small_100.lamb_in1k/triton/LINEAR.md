# MobileNetV3 classifier Linear

2026-10-06：`linear_m1_n1000_k1024` 已在 **patch6 / FPGA7 上 PASS**。
8 轮原始 C oracle 检查 8000 个输出，`errors=0`，NR runtime PASS。
这是软件编译路径的修复；patch6 原始 RVV 浮点归约指令的硬件问题没有修复。

## Kernel、布局与尾块

只有一个 Triton kernel：`kernels.py::linear`，一个自动生成的 case：
`linear_m1_n1000_k1024`。

| 参数 | 固定值 |
| --- | --- |
| Input | FP32 `[1,1024]` |
| Weight 物理布局 | FP32 `[1000,1024]`，地址 `Weight[n*1024+k]` |
| Bias / Output | `[1000]` / `[1,1000]` |
| M / N / K | 1 / 1000 / 1024 |
| BM / BN / BK | 1 / 16 / 1024 |
| grid | `[1,63,1]` |

`Out[m,n] = Bias[n] + sum_k(Input[m,k] * Weight[n,k])`。
`tl.dot(..., input_precision="ieee")` 和 bias 保持在同一个 Triton kernel 内。
kernel、case 参数、C oracle 和误差容限在本次修复中均未改变。

最后一块权重起点为 `N-BN=984`。program 61 读取 976…991，只写 976…983；
program 62 读取并写入 984…999。读取可重叠，写入严格互斥。
Host 实际逆序调用全部 63 个 program，确认每个输出恰有一个 owner；
四个 buffer 末端的 PROT_NONE 页面检查越界。

## patch6 修复

旧 `matmul-transpose-b-vectorization-decode` 沿 K 向量化，最后使用
`vfredusum.vs`。patch6 最小测试中，零输入、零初值的 `vfredusum.vs`
和 `vfredosum.vs` 都产生非零结果。旧 Linear 有 3989 处不匹配，最大绝对误差
约 256.07；这些额外误差的具体 RTL 原因尚未确定。

现在使用和 pointwise Conv 相同的标准 triton-riscv tile 物化与 Buddy
`--matmul-vectorization='vector-size=16 vector-type=fixed'`：

1. Triton AST → TTIR，保留真实 `tt.dot`，拒绝 `tt.trans`。
2. triton-riscv `--triton-to-linalg-experimental` → `linalg.matmul`。
3. 每个 program 从原始 `[N,K]` 读取局部 `[BK,BN]` 权重 tile。
   当前权重 tile 为 **64 KiB**，输入 tile 为 4 KiB，bias 和 accumulator 各 64 B。
   这些是编译器在 kernel 内生成的栈缓冲区；没有完整权重预转置、调用端 packing、
   独立 transpose case 或手写算术 MLIR。代价是增加局部搬运和栈空间。
4. Buddy 沿 N 向量化：16 个 lane 分别累加独立输出，生成 `vfmacc.vf`，
   不再需要横向浮点归约。
5. Buddy → LLVM → RISC-V → ELF 指令审计 → BIN → patch6 FPGA。

构建拒绝非预期大小的 Linear 栈分配、堆分配、显式 transpose、残留 matmul、
缺失 vector FMA，以及重新出现的 `vector.reduction` / `vfred[uo]sum`。
没有放宽共享 ISA 审计。Qwen、共享 runtime 源码和其他算子实现均未修改。

## 独立验证

C oracle 使用 FP64 `m/n/k` 循环，直接读取 `Weight[n*K+k]`。
8 轮覆盖 bias-only（含 ±0）、K 首/中/末 basis、正负非二进制精确值、
强抵消和仅最后输出通道非零，并检查 guard 与输入/权重/bias 不变。
容限保持原定义，没有增大容限或跳过 trial：

```text
u = 2^-24
gamma = ((K+2)*u) / (1-(K+2)*u)
bound = gamma * (sum_k(abs(double(X)*double(Weight))) + abs(double(Bias)))
        + (K+2)*2^-149
abs(actual - FP64_reference) <= bound
```

K+2 为乘法、累加、bias 及参考舍入保留余量；绝对乘积之和用于处理抵消。

| 检查 | 结果 | 最大绝对误差 |
| --- | --- | ---: |
| Host C oracle，8 轮 | PASS | 1.07793882114e-5 |
| NR LLVM 在 Host 上运行同一 C oracle | PASS | 1.07793882114e-5 |
| Host / NR LLVM 各 16 轮 FP64、ATen 对照 | PASS | 0.0418633463 |
| 两份 LLVM 各 63 个 program 的写入边界 | PASS | 每个输出恰有一个 owner |
| NR ELF 指令审计 | PASS | 无横向浮点归约 |
| patch6 / FPGA7，原始 C oracle 8 轮 | **PASS** | **8.70843905432e-6** |

随机对照包含 2^-4、1、2^4 的输入/权重尺度，最大 `error/bound` 为
0.00385424937。板上使用 11.0592 MHz、115200 8N1、UART 分频 6；
上传镜像和 DDR 读回 SHA256 一致。正式构建重链接出的 patch6 镜像与
实际测试镜像逐字节一致。

## 构建和上板

在仓库根目录执行：

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
P=/home/zhangwenji/triton-riscv/.venv/bin/python
M=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k
"$P" -B "$M/triton/build.py" --family=linear --host --nr
```

通用 NR 构建仍使用共享 runtime 原有 UART 分频 8。本次 patch6 将相同
算子对象与已有的 UART 分频 6 runtime 对象重链接，未修改共享 UART 源码。
重链接和审计命令、ELF/BIN 及哈希记录于：

`examples/FPGA-BOSCAME/build/fpga-runs/linear-patch6-fix-20261006-221018/patch6/build.json`

实际复用已有仓库 runner 的 patch6 适配副本，保留先开串口后加载、镜像校验、
DDR 读回和占用检查。具备 SSH 认证时复跑：

```bash
R=examples/FPGA-BOSCAME/build/fpga-runs
python3 -B "$R/mobilenet-patch6-20261006-133234/runner/tools/fpga_run.py" \
  "$R/linear-patch6-fix-20261006-221018/patch6/linear_m1_n1000_k1024.padded.bin" \
  --fpga=7 --remote-dir=Desktop/fpga-tester-ISCAS-patch6-codex-20261006 \
  --capture-seconds=1800 --startup-timeout=300 \
  '--completion-marker=[nr] RA returned:'
```

共享交付缺少 `test/pad_to_fixed.sh`，因此使用独立的 1 MiB 补零文件，
原始 `.bin` 完整保留。工作目录的 Makefile/test/user_script 来自 patch6，
`hw.dat` 链接到 `/bitstream/nanhu-ra/nr-patch6/hw.dat`。

## 产物与证据

以下编译产物相对于 `triton/build/linear_m1_n1000_k1024/`：

| 阶段 | 文件 |
| --- | --- |
| TTIR / Linalg | `kernel.ttir` / `kernel.linalg.mlir` |
| NR vector IR | `nr/vectorized.mlir` |
| LLVM dialect / LLVM IR | `nr/kernel.llvm.mlir` / `nr/kernel.ll` |
| 汇编 / ELF 审计 | `nr/kernel.nr.S` / `nr/elf-audit.json` |
| 默认 runtime ELF/BIN | `nr/linear_m1_n1000_k1024.{elf,bin}` |

[随仓库保存的验证记录](../validation/fpga-20261006/README.md)
包含镜像哈希、原始 UART 和逐项校验结果；patch6 ELF/BIN 保留在上述本地构建目录。
验证索引在 `../validation/linear/{host,nr,fpga,status}.json`。
历史 FAIL 日志保留；原始 RVV 归约最小复现仍为 FAIL，没有替换它来登记硬件 PASS。

**Linear 软件修复：COMPLETE**。本次未重测其他全部算子或完整模型。
