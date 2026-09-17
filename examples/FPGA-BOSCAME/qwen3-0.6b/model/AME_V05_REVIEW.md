# AME v0.5 算子优化机会核对

2026-09-17。本文保存实施前的规范/源码分析；下文性能预测仍是候选分析。
实施后的独立板测、编译证据与模型回归见 [optimization/RESULTS.md](optimization/RESULTS.md)。

## 文档与现有硬件的关系

本地参考文件（不随 Git 上传）：

| 文件 | SHA256 |
|---|---|
| `references/RAV0.5_FPGA_ALPHA_260908 硬件文档.pdf` | `d3975c548c96337c9eb1ae996f05925cc0fa1f9966464c08e43494515c504805` |
| `references/AME v0.5 软件使用说明（2026-9-6版本）.pdf` | `60720fa304f8e40ce8cf2b89fe5fc1fd0269dda4ebb4182758b8a7653995e820` |

硬件文档 p1 标明 `nanhu-fpga-aps` 的 `feat/AME`，commit
`6c4d6dc1ab8197cac1a7ef2cb0e8a61fb9513402`；软件文档 p1 标明
`ame-core-v0p5@ef073bb192243952bce1376643fd0491b20a3495`。

最新通过的单层运行 `run-550440b96df24b2c` 的
`validation/board/review/model-1l-intermediates-no-profile/run/uvhs.log`
第 79/89 行引用同名 `RAV0.5_FPGA_ALPHA_260908` 工程的 `hw.dat` 和
`MOD_b0_f0.bin`。本次 SSH 只读查询亦确认允许目录中的 `Makefile` 指向该工程。
这是文档适用于当前平台的强证据；尚未核对 PDI 哈希或板上 RTL commit 标识。

硬件文档 p2–3：INT8 为 512 MAC/cycle，原生 task 为 16×32，最大 M/N/K
均为 64；64³ GEMM 拆成 8 个原生 task。AME 直接访问 DDR，AXI 数据宽 512 bit，
最多 16 outstanding。硬件文档报告 INT8/FP8 的 64³ 板级闭环；软件文档 p9
另外给出 ChiselSim/VCS 验证边界。二者都不提供当前 Qwen shape 的性能实测。

按工程配置的 14.7456 MHz，1024 operations/cycle 对应理论峰值约
15.10 GOPS；文档中的 1.024 TOPS 是 1 GHz 下的换算。M=1、访存和指令调度
会限制利用率，不能拿峰值直接推算 decode 时间。

## 已测基线和生成代码证据

使用已通过数值验收的 `run-390dfc362f234ec6`：
`validation/board/review/model-1l-native/kernel-profile-verification.json`。
每图 39 次静态库 kernel 调用；8 步 decode 的平均图计算为 291,090,841 cycles。

| kernel | 平均 cycles/step | 按配置时钟折算 |
|---|---:|---:|
| lm_head `matmul_1x151936x1024` | 158,617,320 | 10.76 s |
| lm_head `dequantize_1x151936` | 37,597,919 | 2.55 s |

二者合计约占单层图计算 67.4%。这是带 profiler 的另一份镜像，不能与最新
关闭 profiler、保留中间值探针的镜像混为同一性能基线；秒数也不是独立墙钟测量。
kernel 计时包含 ABI adapter 和完成同步，不能全归为硬件 MMA 用时。

归档 `build/model-lib/evidence/` 的两个 `kernel.s` 与当前 Triton build 中
相应 LLVM 输出逐字节相同。matmul 的 constexpr 是 BM=1、BN=16、BK=1024，
grid 为 1×9496×1；Buddy NR lowering 再把 K 分成 16 个 64-byte 块。

当前 `kernel.nr.S` 的单一路径、16 次 K 循环和 9496 次 program 调用给出以下
静态执行次数估算（不是硬件计数器测量，不含外层 adapter/profiler）：

| 指令 | 每 program | 每次 lm_head |
|---|---:|---:|
| AME word | 71 | 674,216 |
| `fence` | 144 | 1,367,424 |
| `sd` | 136 | 1,291,456 |
| `ld` | 142 | 1,348,432 |

`sd/ld` 包含函数自身和 AME 固定寄存器转接，并非全部可以删除。上述数量也不能
直接换算成节省 cycles：DDR、流水及等待成本不同。

## 优化候选

### 1. 反量化融合和真正的 RVV 算术，优先实施

`../triton/kernels.py::dequantize` 当前通过通用扁平索引计算
`Row[index // COLS]`、`Column[index % COLS]`。实际汇编有索引数组、标量除法、
多个 512-byte 栈临时；转换用 `fcvt.s.w`，两次乘法用 `fmul.s`。
目前向量化的是部分拷贝，不能因此声称整个反量化已经向量化。

lm_head 的 ROWS=1 且 151936 可被 BLOCK=128 整除。可在 Triton 中特化为
标量加载 Row[0]、连续 Column/输入/输出，保持
`(float(acc) * row_scale) * column_scale` 的既有运算顺序；在 Buddy
结构化 IR 阶段融合/向量化，生成 `vfcvt.f.x.v`、`vfmul.vf`、`vfmul.vv`。
这些指令已有 e32/m1/VL16 板测 PASS，记录见
`../validation/rvv-capabilities-expanded/uart.raw.log`。

先单独优化此核；之后再考虑与 matmul epilogue 融合，减少 INT32 临时缓冲的
写回/读回。融合时要保留完整 logits 和当前比较点，不能以减少验证范围换速度。
quantize 的转换/归约/舍入也有类似优化空间，但须保持原来的舍入和 clamp 语义。

### 2. 按真实 GPR 编码 MLS 和 tile 配置，先保留屏障

软件文档 p2、p4、p8–9 明确配置/MLS 支持 x0..x31，base/stride 字段各 5 bit。
当前 `../../tools/ame_to_word.py` 将它们搬到固定寄存器，再保存/恢复原值。
可直接使用 LLVM 已分配的寄存器编码，去掉这层转接。

首轮只改 MLS 和 `msettile*`，保留 mtype、fence、同步顺序。`msettype`
当前有保存 15 个 caller-saved GPR 的额外防护，其 clobber 语义应另做实验。
编码审核 `nr_isa.py` 和 ELF 审核需要同步支持相应新形式。

验证高 GPR、rd==rs1、base/stride 重叠、sp/zero、padding stride、非对齐和
partial tile；不仅要查机器码，还要上板逐元素核对。此项不要求改变量化模型。

### 3. N 分块从 16 扩到 32/64，同时修改前端和 NR lowering

`../triton/cases.py` 把 BN 固定为 16；
`NrMatmulToBOSCAMELowering` 中 stepN、columns 也固定为 16。
不能仅改 Triton BN，否则 Buddy 仍会拆回 N16。

同一个 lm_head 改 N32/N64 后，program 数分别为 4748/2374，对应减少重复
配置、A load、wrapper 和 C 临时处理。N32 也更贴近文档的 16×32 原生 task。
K 硬件分块仍为 64，累加器保持跨 K 常驻，INT32 累加语义不变。

program 数减少 4 倍不等于运行时间减少 4 倍：lm_head 仍需读取
155,582,464 bytes（148.375 MiB）的 INT8 权重，M=1 利用率限制仍在。
现有模型权重已是物理 [N,K]，无需新增全矩阵转置；代码中旧
`qwen3-fpga` 的 1A8B/2A4B 路线不属于当前 `nr-fpga`，不能直接打开使用。

### 4. 简化 mtype、fence 和 synthetic resync，逐项单独验证

软件文档 p3：`MQMA.B.MM` 自带 INT8 mode，reset mtype=0 足够；MLS 自带
width。p6：每条操作 completion 表示自身完成，不要求猜周期 delay。
当前每 tile 仍有 accumulator/INT8/accumulator 的三次 mtype 设置，且每条
AME、每条 RVV memory 前后都插 fence。

`../../common/nr/ame_sync.c::ame_fence` 还发射 10 条 AME 指令完成微型 resync，
不是普通 fence。它用于 profiler 和中间值探针，是值得隔离测量的开销，也是
profiler 停顿的排查候选，尚不能认定它导致停顿。

文档 p6 同时明确 AME non-coherent，fence 不替代 cache clean/invalidate。
应先测试 CPU→AME、RVV→AME、AME→CPU/RVV、连续缓冲复用的完成及可见性，
再分别移除重复配置、缩减屏障或替换 resync。不能把三项一起删除。
还要核对 backend 依赖 msettype 判断 FPGA 编码的行为，保留显式 target
属性，防止优化后意外生成 GEM5 的另一套编码。

### 5. FP8 和并行调度放到后续

FP8 E4M3→FP32 是新文档支持的能力，但当前 kernel/白名单/量化参考只支持
INT8→INT32。FP8 会改变输入量化和累加误差，不是 W8A8 的透明替换；需独立
host FP8 reference、板测和完整模型精度验收。不能承诺它优于 INT8。

独立 AXI master 和 16 outstanding 给访存调度提供可能性，但软件文档 p9
明确未验收 AME/RVV 并行同步。不能据此直接开启异步重叠。当前不需要恢复
已被 v0.5 废弃的 SPM/mspml/mspms 路线。

## 推荐验证顺序

1. 使用现有 checkpoint 作回退基线，固定输入、量化规则和比较范围。
2. 反量化 Triton 特化＋Buddy RVV：检查 IR/最终机器码、host 及 FPGA 逐元素结果，
   再记录独立 kernel cycles。
3. AME 动态 GPR 编码，保留其余行为：先小型编码/数据 probe，再 lm_head 对照。
4. 分别测 N16/N32/N64；覆盖 M=1/16、K64/full K、多次累加、非零 C 和 tail。
5. 再隔离 mtype/completion/cache/resync，保留重复运行与固定布局的对照。
6. 只有 kernel 数值验收后才链接回 1 层固定 16+8，重跑中间值、完整 logits/KV；
   最后进入 4/28 层。未通过的实验不能覆盖已验收默认路径。

所有修改仍走 Triton → triton-riscv → MLIR → Buddy → static library；adapter
只转换 ABI。当前没有已测加速倍数，也没有新的 28 层 FPGA 结论。
