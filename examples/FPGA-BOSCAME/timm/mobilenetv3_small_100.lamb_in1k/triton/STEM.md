# MobileNetV3 stem：NCHW FP32 dense Conv2D

> 2026-10-06 后续上板结果见 [逐 case 验证记录](../validation/fpga-20261006/README.md)。下文的 `NOT EXECUTED` 为初次构建时的历史状态；patch5 与 patch6 的结果分别记录。

实际目录沿用此前合并后的 `examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/`。
本次仅新增一个 Triton kernel `kernels.py::conv_stem`，一个静态 specialization：

| case | Input NCHW | Weight OIHW | Bias | Output NCHW | M / N / K |
| --- | --- | --- | --- | --- | --- |
| conv_stem_cin3_cout16_h224_k3_s2_p1 | [1,3,224,224] | [16,3,3,3] | [16] | [1,16,112,112] | 12544 / 16 / 27 |

kernel=3×3、stride=2×2、padding=1×1、dilation=1、groups=1，输入/权重/输出为连续 FP32。
Bias 接受离线 BN folding 后的值，kernel 内完成 bias 加法。
没有添加其他普通 dense spatial Conv 的配置，也没有修改已有算子或 Qwen 实现。

## Tile、布局和 padding

BM=16、BN=16、BK=32，grid=`[784,1,1]`。
每个 program 负责 16 个输出空间位置、16 个输出通道，按如下公式直接取原张量：

```text
m  = oh*OW + ow
k  = (ic*KH + kh)*KW + kw
ih = oh*2 + kh - 1
iw = ow*2 + kw - 1
A[m,k] = X[(ic*224 + ih)*224 + iw]，越过 H/W 边界则为 0
B[k,n] = Weight[n*27 + k]
Out[n*12544 + m] = dot(A,B)[m,n] + Bias[n]
```

Input mask 同时检查输出空间范围、K<27 和 ih/iw 的上下界；Weight mask 检查
输出通道和 K<27；Bias 和输出 store 也分别检查通道、空间范围。K 的第 27–31
lane 补零，不能读取第四个输入通道或下一输出通道的权重来凑足 BK。
M/N 在唯一 case 中恰好整除 tile，K=27 不整除 BK=32。

采用真正的 FP32 `tl.dot(..., input_precision="ieee")`，不生成完整 im2col buffer。
编译器将当前 tile 的 gather、mask、dot 所需数据暂存于栈上；bufferized MLIR 中
单个 allocation 最大为 512 元素，包括 FP32、i32 和 i1 临时数组。
LLVM 中没有堆分配。没有独立 transpose、padding、bias 或 matmul case。

注意这个 stride/padding 组合：输出 (0,0) 的输入有效区是 [0:2,0:2]；最后输出
(111,111) 读取 [221:224,221:224]，触及最后一行/列但不读取 224。
padding=1 并不意味着四个输出角的有效像素数量相同。

## 真实编译路径

```text
Triton ASTSource / CPUBackend
  → kernel.ttir（tt.dot）
  → triton-riscv --triton-to-linalg-experimental
  → kernel.linalg.mlir（linalg.matmul + masked tile loads/stores）
  → Buddy bufferization
  → NR: --matmul-vectorization='vector-size=16 vector-type=fixed'
  → vector.fma → LLVM dialect → LLVM IR
  → RISC-V RVV FP32 vfmacc → shared NR runtime → ELF/BIN
```

NR 复用现有 Qwen/pointwise 的 FP32 fixed-vector matmul 路径。
`STEM_FP32_PASS`、`STEM_RVV_FLAGS` 在 `common.mk` 中配置。
Exporter 验证真实 `tt.dot` 和 `linalg.matmul`；NR build 验证 `vector.fma` 和
实际 `vfmacc` 汇编指令，并执行仓库现有 ISA 审计。没有手写计算 MLIR。
本次使用本机已安装工具链，实际路径、版本与 SHA-256 见 `validation/conv-stem/toolchain.json`。

## 独立验证和误差界

`launch_conv_stem.c.in` 自动生成该 case 的 `launch.c`，batch=1 后使用直接的
oc/oh/ow/ic/kh/kw 六重 C loop，按 NCHW/OIHW 索引读取，显式跳过 padding 像素，
以 double 累加，最后加 bias。不调用矩阵乘法库，不用 Triton 结果生成 reference。

12 轮 C 测试覆盖 ±0/bias-only、通道/空间 ramp、首/末输入通道、全部 27 个
one-hot 权重位置、四角脉冲、边界条带、checkerboard 抵消、仅最后输出通道非零、
非二进制精确正负值，以及全 1 输入/权重的有效像素计数。
Host 在四个 buffer 的实际尾部放置 PROT_NONE 页，并检查前缀 canary、
X/Weight/Bias 逐位不变；NR 使用同一 C oracle 和前后 canary。

误差界在测试前由 K=27 和 FP32 精度确定，不根据实测误差放宽：

```text
u = 2^-24
gamma = (29*u)/(1-29*u)
bound = gamma * (sum_valid(abs(double(X)*double(Weight))) + abs(double(Bias)))
        + 29*2^-149
abs(actual - double_reference) <= bound
```

每项最多经历一次 FP32 乘法、26 次加法和一次 bias 加法，再留一次舍入余量，
覆盖 double reference 与误差界求值误差。这个界也覆盖更少舍入的 FMA 累加。
零 padding 和补齐的 K lane 不增加误差；容限按 K=27 而非 BK=32 计算。
分析针对有限输入、round-to-nearest、无中间溢出，绝对项为下溢留余量。

`verify_conv_stem.py` 另执行固定 seed=0 的 39 轮测试：12 种模式/随机尺度，
以及 27 个逐位置 one-hot 权重。对照 FP64 conv2d 和 FP32
`torch.ops.aten.convolution.default`，两种 FP32 结果分别满足 FP64 界，彼此差异
不超过两倍界。四角和内部的 5 个 tile 还单独调用真实编译入口，检查 NCHW
输出范围和其他位置的 sentinel。普通 Host LLVM 和 NR 向量化 LLVM 分别执行。

| 实际验证 | 结果 |
| --- | --- |
| 独立 C oracle，12 轮 | PASS |
| FP64 / ATen 对照，39 轮、7,827,456 个输出样本 | PASS |
| NR 向量化 LLVM 的相同 C + FP64 / ATen 对照 | PASS |
| 逐 tile 写入检查 | Host 5 次、NR-vector Host 5 次 PASS |
| C 最大绝对误差 / 最大 error-bound 比值 | 1.30600392545e-6 / 0.152765653 |
| FP64 压力测试最大绝对误差 / 最大 error-bound 比值 | 0.00154868754 / 0.156778284 |
| NR ELF/BIN 构建、ISA 审计 | PASS |
| FPGA 实际运行 | **NOT EXECUTED** |

压力测试包含输入和权重同时乘 16 的样本，其绝对误差不能直接与 C 小尺度
样本比较。NR 向量 LLVM 在 Host 的通过不代表 FPGA 已通过。

原有 87 个 case inventory、9 个 kernel AST 和生成目录保持不变。
5 个代表性 Host 回归（ReLU、Linear、depthwise、两个 pointwise）全部 PASS；
所选 Linear/pointwise 的 NR LLVM 与修改前逐字节一致。详见 `regression.json`。

## 命令和产物

在仓库根目录执行：

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
P=/home/zhangwenji/triton-riscv/.venv/bin/python
M=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k
CASE=conv_stem_cin3_cout16_h224_k3_s2_p1
B="$M/triton/build/$CASE"

"$P" "$M/triton/cases.py" --family=conv_stem --generate
"$P" "$M/triton/build.py" --family=conv_stem --host
"$P" "$M/triton/build.py" --family=conv_stem --nr
```

`--nr` 必须先验证本 family 所有 case 的 Host PASS，才开始 NR 构建。
也可使用 `make -C "$M/triton" FAMILY=conv_stem TRITON_PYTHON="$P" check` / `all`。

| 产物 | 路径 |
| --- | --- |
| 自动生成元数据 / C oracle | `$M/$CASE/metadata.json` / `$M/$CASE/launch.c` |
| TTIR | `$B/kernel.ttir` |
| Linalg MLIR | `$B/kernel.linalg.mlir` |
| NR 向量化 MLIR | `$B/nr/vectorized.mlir` |
| LLVM dialect / LLVM IR | `$B/nr/kernel.llvm.mlir` / `$B/nr/kernel.ll` |
| RISC-V assembly | `$B/nr/kernel.nr.S` |
| ELF，46,264 bytes | `$B/nr/conv_stem_cin3_cout16_h224_k3_s2_p1.elf` |
| BIN，85,466 bytes | `$B/nr/conv_stem_cin3_cout16_h224_k3_s2_p1.bin` |
| Host C / NR-vector Host C 日志 | `$B/host/output.log` / `$B/nr/vector-host.log` |
| ELF 指令审计 | `$B/nr/elf-audit.json` |

报告目录为 `validation/conv-stem/`，包含 host/nr/regression/toolchain/inventory/
model-coverage/fpga/status JSON。逐文件 SHA-256 和最终状态见
[`status.json`](../validation/conv-stem/status.json)。

2026-09-29 已以普通用户 hjuser 登录成功（未使用 sudo），通过原 `fpga_run.sh`
依次尝试 FPGA5、4、6、7，并重试 FPGA5、在检查占用后重试 FPGA4；每次镜像上传均成功。实际结果：

| 板卡 / 尝试 | 阻塞点 | 程序执行 |
| --- | --- | --- |
| FPGA5 首次 | 工作目录已有 UVHS 会话 PID 30258 | NOT EXECUTED |
| FPGA4 | `/dev/FPGA4` 被 PID 31966 占用 | NOT EXECUTED |
| FPGA6 | `load_db ... b1.f2` 报外部中断，加载终止；UART 为 0 bytes | NOT EXECUTED |
| FPGA7 | `/dev/FPGA7` 被 PID 27512 占用 | NOT EXECUTED |
| FPGA5 重试 | UVHS 报 `Daemon is not start! Failed to connect to system` | NOT EXECUTED |
| FPGA4 检查后重试 | 串口已空闲、原进程已退出；UVHS daemon 仍不可连接 | NOT EXECUTED |
| FPGA4 再次重试 | 新 Qwen2.5 UART 采集 PID 38761 占用串口，未中止该进程 | NOT EXECUTED |
| FPGA5 再次重试 | 13:24:43 UVHS daemon 仍不可连接 | NOT EXECUTED |
| FPGA4 最新重试 | 13:32:24 无串口占用阻塞，UVHS daemon 仍不可连接 | NOT EXECUTED |
| FPGA4 手动 scp/minicom/make | 原始 BIN 上传和 1 MiB 补零成功；13:36:21 UVHS daemon 仍不可连接，UART 0 bytes | NOT EXECUTED |

按用户后续指示还执行了手动流程，未调用 `fpga_run.sh`：先 scp 上传原始 BIN，
校验 SHA-256，再打开 minicom（115200/8N1、关闭流控），另一个终端执行
`make uv_run4 test=test/conv_stem_manual_20260929.bin`。生成的新补零文件原始前缀
及尾部全零均已校验；make 最终退出码为 2，UVHS daemon 仍不可连接。
详见 [`manual-20260929/result.json`](../validation/conv-stem/manual-20260929/result.json)。

本次未得到 kernel 的数值 PASS/FAIL；所有失败均发生在程序启动之前。
最新阻塞条件为 UVHS daemon 不可连接，不能通过修改 kernel 解决。
日志、命令、BIN hash 见 [`fpga-ports.json`](../validation/conv-stem/fpga-ports.json)。
本地 ELF/BIN 保留；平台恢复且板卡空闲后可复用下列命令：

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  "$B/nr/$CASE.bin" --fpga=5 \
  --capture-seconds=120 --completion-marker='[nr] RA returned:'
```

也可在已配置认证后执行 `"$P" "$M/triton/build.py" --family=conv_stem --run`。
只有原脚本成功退出，并实际捕获 case 数值 PASS 与 `[nr] RA returned: PASS`，
才记录 FPGA PASS。

本地实现、Host 验证和 NR 构建：**COMPLETE**。FPGA：**NOT EXECUTED**。
包含实际 FPGA 验证的本次任务：**INCOMPLETE**，阻塞于 UVHS 服务不可连接及板卡/串口占用。
