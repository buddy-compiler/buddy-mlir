# SE channel broadcast multiply：7 个静态 Triton cases

> 2026-10-06 后续上板结果见 [逐 case 验证记录](../validation/fpga-20261006/README.md)。下文的 `NOT EXECUTED` 为初次构建时的历史状态；patch5 与 patch6 的结果分别记录。

实现位于此前合并后的 `examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/`。
本次仅新增 `kernels.py::se_mul` 一个通用 Triton kernel，执行：

```text
X:     [1,C,H,W]
Scale: [1,C,1,1]
Out:   [1,C,H,W]
Out[0,c,h,w] = X[0,c,h,w] * Scale[0,c,0,0]
```

模型 profile 保持 batch=1、输入 `[1,3,224,224]`、NCHW、FP32、eval、输出 `[1,1000]`。
输入和输出均为连续张量，Scale 的实际存储只有 C 个 FP32 元素。

## NCHW 寻址与 specialization

采用二维 grid `[ceil(H*W/BLOCK), C, 1]`，BLOCK=128：

- `channel = program_id(1)`；
- `spatial = program_id(0)*BLOCK + arange(0,BLOCK)`；
- `index = channel*(H*W) + spatial`；
- 每个 program 标量读取 `Scale[channel]`，在本 tile 内广播相乘。

X/Out 的 load/store 使用 `spatial < H*W` mask，kernel 还包含通道范围检查。
每个通道末尾独立处理不足一个 tile 的空间区域，不能只判断整个张量的 COUNT。
没有将 Scale 扩展为 COUNT 个元素，ABI adapter 只传递指针描述符和 grid/pid。
本类无卷积权重或 OIHW，也没有空间 padding；mask 只处理 tile 的无效 lane。

| case | X / Out NCHW | Scale NCHW | COUNT | grid | 每个通道尾部有效元素 |
| --- | --- | --- | ---: | --- | ---: |
| se_mul_c16_h56_w56 | [1,16,56,56] | [1,16,1,1] | 50176 | [25,16,1] | 64 |
| se_mul_c96_h14_w14 | [1,96,14,14] | [1,96,1,1] | 18816 | [2,96,1] | 68 |
| se_mul_c240_h14_w14 | [1,240,14,14] | [1,240,1,1] | 47040 | [2,240,1] | 68 |
| se_mul_c120_h14_w14 | [1,120,14,14] | [1,120,1,1] | 23520 | [2,120,1] | 68 |
| se_mul_c144_h14_w14 | [1,144,14,14] | [1,144,1,1] | 28224 | [2,144,1] | 68 |
| se_mul_c288_h7_w7 | [1,288,7,7] | [1,288,1,1] | 14112 | [1,288,1] | 49 |
| se_mul_c576_h7_w7 | [1,576,7,7] | [1,576,1,1] | 28224 | [1,576,1] | 49 |

清单与 `model.json` 中 9 次 SE `aten.mul.Tensor` 调用的 7 组去重 X/Scale shape 一致。
`(144,14,14)` 与 `(576,7,7)` 的 COUNT 相同，但 SPATIAL、C 和 Scale shape 不同，
因此本类按 `(C,H,W)` 保留两个独立 case。纯 elementwise family 的 COUNT 去重行为不变。

`cases.py --family=se_mul --generate` 从统一 inventory 和 `launch_se_mul.c.in`
生成全部 7 个目录，各含 `launch.c`、`metadata.json`、`makefile`；正常 build 也自动生成。
kernel 的 constexpr 为 COUNT、SPATIAL、BLOCK，不复制 kernel。

## 独立数值验证

每个 `launch.c` 用独立的 C 三重 `c/h/w` 循环，在 kernel 调用前计算：

```c
int index = (c * HEIGHT + h) * WIDTH + w;
reference[index] = x.data[index] * scale.data[c];
```

四轮确定性输入分别覆盖：X=1 配合各通道不同的 Scale；逆序通道 Scale；
正负 X/Scale、±0、±1；仅最后一个通道 Scale=1、其余通道 Scale=0。
这样可以区分 NCHW/NHWC、固定单个 Scale、错误通道步长及同 shape 逐元素乘法。
有限值采用精确 FP32 数值比较，允许正负零相等；X 和 Scale 的不可变性按位检查。

Host 的 `X[COUNT]`、`Out[COUNT]`、`Scale[C]` 分别紧邻 PROT_NONE 页，
可发现尾部 load/store 越界及错误地把 Scale 当作 COUNT 长数组读取。
前方保留 canary；NR 在三个缓冲区前后各放置 128 个 guard。
Scale 的有效元素始终为 C，guard 不属于其有效张量。

`verify_se_mul.py` 将相同 Host LLVM IR 与 ABI adapter 链接成动态库，
通过 ctypes 调用真实编译产物，并与 `torch.ops.aten.mul.Tensor(X, Scale)` 对照。
PyTorch 的 Scale 同样只分配 `[1,C,1,1]`，不 materialize 成 X 的 shape。
每个 case 使用 seed=0 的 16 轮输入：4 轮通道布局/零值模式和 12 轮随机模式，
随机 Scale 覆盖 `[0,1)` 或 `[-2,2)`。该检查补充独立 C oracle；任一失败立即停止。

实际结果：

- 独立 C Host 检查：**7/7 PASS**，最大绝对误差均为 **0**。
- PyTorch 2.10.0+cpu 广播对照：**7/7 PASS**，**3,361,792** 个元素样本，最大绝对误差 **0**。
- NR ELF/BIN 构建及 ELF 指令审计：**7/7 PASS**。
- 已有 family Host 回归：residual_add **4/4**、ReLU **11/11**、hardsigmoid **7/7**、
  hardswish **9/9 PASS**，共 31 项；此前的一维 adapter 输出逐字节不变。
- FPGA：**NOT EXECUTED**，沿用用户暂停上板的约定。
- 本次状态：**COMPLETE**，范围为 SE multiply 实现、Host 验证和 NR 二进制构建。

证据：[host.json](../validation/se-mul/host.json)、[nr.json](../validation/se-mul/nr.json)、
[regression.json](../validation/se-mul/regression.json)、[status.json](../validation/se-mul/status.json)。
回归产物放在 `triton/build/regression-se-mul/`，此前各类的产物和报告保留。

## 实际构建链与命令

Triton AST → TTIR → triton-riscv `triton-to-linalg-experimental` → Linalg
→ Buddy bufferization/lowering → LLVM → RISC-V → NR ELF/BIN。
TTIR 中包含二维 program ID、Scale 的标量 load 和广播；Linalg 中存在真实
`linalg.generic` / `arith.mulf`，NR 汇编包含 `fmul.s`。
没有手写 `kernel.mlir`。Qwen3、公共 runtime 和上板脚本未修改。

仓库根目录运行本机已经验证的命令：

```bash
export TRITON_SHARED_OPT_PATH=/home/zhangwenji/triton-riscv/backend/bin/triton-shared-opt
export BUDDY_BIN=/home/zhangwenji/buddy-mlir/build/bin
export LLVM_BIN=/home/zhangwenji/buddy-mlir/llvm/build/bin
export RISCV_LD='ld.lld -m elf64lriscv'
export PYTHONSAFEPATH=1
MOBILENET_PYTHON=/home/zhangwenji/triton-riscv/.venv/bin/python
MOBILENET_TRITON=examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton

"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=se_mul --host
"$MOBILENET_PYTHON" "$MOBILENET_TRITON/build.py" --family=se_mul --nr
```

`--nr` 先验收全部 7 个 Host case，再构建 NR runtime 和二进制；仅当源码、
工具及产物摘要一致时复用已有 PASS 证据。Make 入口如下：

```bash
make -C "$MOBILENET_TRITON" FAMILY=se_mul check TRITON_PYTHON="$MOBILENET_PYTHON"
make -C "$MOBILENET_TRITON" FAMILY=se_mul all TRITON_PYTHON="$MOBILENET_PYTHON"
```

使用本机已安装的外部 Triton/Buddy/LLVM，实际路径、版本、SHA-256 见
[toolchain.json](../validation/se-mul/toolchain.json)，不声称匹配 Qwen3 的锁定工具链版本。

## 产物及后续上板

每个 case 的产物均在 `triton/build/<case>/` 下：

| 产物 | 相对路径 |
| --- | --- |
| TTIR | `kernel.ttir` |
| Linalg MLIR | `kernel.linalg.mlir` |
| 前端记录 / grid ABI adapter | `frontend.json`、`adapter.c` |
| Host LLVM / C oracle / 日志 | `host/kernel.ll`、`host/check`、`host/output.log` |
| PyTorch 对照库 / 报告 | `host/pytorch-check.so`、`host/pytorch.json` |
| NR LLVM dialect / LLVM IR | `nr/kernel.llvm.mlir`、`nr/kernel.ll` |
| RISC-V 汇编 | `nr/kernel.nr.S` |
| ELF / BIN | `nr/<case>.elf`、`nr/<case>.bin` |
| ELF 审计 | `nr/elf-audit.json` |

例如 `se_mul_c576_h7_w7` 的 BIN 为
`triton/build/se_mul_c576_h7_w7/nr/se_mul_c576_h7_w7.bin`。
以下命令供后续使用，**本次未执行**：

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/timm/mobilenetv3_small_100.lamb_in1k/triton/build/se_mul_c576_h7_w7/nr/se_mul_c576_h7_w7.bin \
  --fpga=5 --capture-seconds=120 --completion-marker='[nr] RA returned:'
```

`build.py --family=se_mul --run` 可依次调用同一个公共脚本。只有脚本成功退出，
同时出现对应 case 的数值 PASS 和 `[nr] RA returned: PASS` 才记录 FPGA PASS。
本次未连接服务器或执行板上程序。
