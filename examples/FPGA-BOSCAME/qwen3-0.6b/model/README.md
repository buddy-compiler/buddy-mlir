# Qwen3-0.6B：Buddy 计算图 + Triton 静态库 → NR FPGA

**完整 28 层 FPGA 16+8 尚未验收；当前按用户要求用预置原始文本，UART RX 暂缓。** 2026-09-17 复核修复了多层
返回结构越界、跨层 KV alias、图/adapter/workspace 错配、临时 heap 不回收、交互 token
错位和 UART mailbox 所有权问题。当前结果与旧记录必须区分；修复细节与构建命令见
[REVIEW.md](REVIEW.md)，机器可读状态见 [validation/evidence-index.json](validation/evidence-index.json)。

`examples/BuddyQwen3` 保持原主机路径。本目录复用公共 Triton kernel、Buddy lowering
与 `common/nr` runtime，模型执行顺序来自实际导入图，C glue 负责 ABI、内存和交互。

## 当前验收状态

| 范围 | 当前结论 | 证据 |
| --- | --- | --- |
| 官方 checkpoint/config | 已核对 28 层、1024 hidden、3072 intermediate、16 Q/8 KV、head_dim=128、RoPE θ=1e6、vocab=151936、ε=1e-6、tied embedding/lm_head | `validation/checkpoint-verification.json` |
| 真实 Buddy 图、算子清单 | 已导入，原始每图 3217 ops，清单未分类项为 0；分类不等于全部替换 | `validation/operator-inventory.json` |
| 模型形状 Triton 单 kernel | 历史 Stage A 11/11 FPGA5 PASS，保留各 run/hash/cycles | `validation/stage-a/model-kernels-fpga.json` |
| **1/4/28 层主机编译图** | 与独立 `triton-host` W8A8 参考的末位完整 logits、逐步有效 KV 均 max/mean error=0；16 prefill + 8 decode | `validation/review-host-{1,4,28}l.json` |
| **1 层 FPGA 完整替换集** | 16 prefill + 8 decode；每步全部 151936 末位 logits 和全部有效 KV，共 27 项比较 max/mean error=0；对照独立 `nr-fpga` 参考 | `validation/board/review/model-1l-full/` |
| 28 层 FPGA | **未验收**；旧镜像存在多层软件缺陷，旧长时间无输出不能解释为“只慢” | `REVIEW.md`、`validation/round-state.json` |
| tokenizer 主机差分 | 5173 编码/regex、26464 NFC、144 template→encoding 等检查通过；完整数字见报告 | `validation/tokenizer-encode-audit.json`、`validation/text-check-audit.json` |
| **tokenizer FPGA** | **8/8** 字面文本 fixture 的模板、编码 ID、增量解码完全一致；包含中文/NFC、两种 thinking 模式；不经过 UART 输入 | `validation/board/review/tokenizer-probe/` |
| 公共 TX 与 SSH 恢复 | 67,584 字节跨 ring 边界逐字节一致，真实断线恢复同一 worker | `validation/board/review/console-wrap-probe/` |
| UART RX / Stage E | FPGA5 探针 host 写入 16 字节，板侧未见接收；**根因待查**，平台脚本有 RX 配置；未完成文本交互推理 | `validation/board/review/uart-rx-probe/`、`transport-review.md` |
| DDR | 19 个分散 64-byte 地址通过，不是全部容量扫测 | `validation/board/review/ddr-address-probe/` |
| 28 层 native-K 固定文本 ELF 内存 | HIGH 持久区 **791.30 MiB / 1,152 MiB**，余 360.70 MiB；LOW scoped heap 独立，28层动态峰值待测 | `validation/review-memory-native-28l-fixed-text.json` |
| 动态有效长度 attention | 4 kernel 共60次长度测试板上通过；接入1层真实图后16+8完整logits/KV仍零误差 | `validation/board/review/attention-position/`、`validation/board/review/model-1l-position/` |

新 1 层板测 `run-015b4fabf4014edc` 的 prefill 为 **1,183,615,186 cycles**，decode
为 **784,100,292–784,135,605 cycles/step**。这些只包围 graph call；NR launch 总计
7,591,649,001 cycles，含验证/输出/初始化。不可用旧 5,515,994,658 cycles 直接推算
本次性能收益，因为图配置、修复、验证范围不同。

动态长度版本 `run-9173a2481e114612` 已完成：prefill **949,799,911 cycles**，decode
**685,809,105–686,344,451 cycles/step**。此镜像额外启用 kernel 计时 wrapper/fence；
真实 LLVM 调用次数与日志吻合，每次 graph call 共40次 kernel。主要瓶颈现已定位到
全容量 K 转置（decode约3.95亿cycles，占graph约58%）。后续 native-K 版本已经去掉这次
转置，1/4/28层主机验证通过；单层 FPGA 的27项 logits/KV 比较均零误差，decode graph
降到 **290,779,337–291,425,808 cycles/step**（含 profiler/fence）。

4层 native-K 固定文本诊断版 `run-b8d3ed3ead854a24` 已通过完整装载、数值、profile、
文本检查，证据为 `validation/board/review/model-4l-native-fixed-text-progress/`。FPGA 自行对
`What is France?` 应用模板、编码16个token、prefill一次、连续decode八次、增量解码九个
预测token。4层截断模型输出不代表28层模型质量。此版逐kernel UART日志计入graph时间，
不能当作纯推理性能。无progress的4层镜像曾超时，原因仍在排查。

另外已确认 PATH 中 Ascend LLD 15 的 RISC-V `--wrap`/relaxation 会使单层诊断镜像函数
入口错位。image builder 现在要求 **LLD >= 20**，支持 `--linker`，记录路径、版本和hash；
新镜像需要重新上板，不能继承旧镜像的数值结论。当前检查点见 [CHECKPOINT.md](CHECKPOINT.md)。

单层中间值检查现已完成：`run-550440b96df24b2c` 以预置原始文本执行16+8，
**414项选定中间张量比较＋27项完整logits/KV比较均零误差**，板上文本处理及严格归档
通过，见 `validation/board/review/model-1l-intermediates-no-profile/`。此版关闭kernel
profiler；此前两次带profiler版本停在prefill之后，具体机制仍未确定。四层不带逐kernel
日志的版本也仍未验收。**目前按用户要求暂停，未启动28层测试。**

## 环境和资源

| 用途 | 环境 |
| --- | --- |
| Buddy Python 导入/JIT | Python 3.11，与 `build-python/python_packages` bindings 相同 ABI |
| Triton 前端 | `boscame`（Python 3.12，已构建的 `thirdparty/triton-riscv`） |
| Buddy / LLVM 工具 | `build-migrate/bin`、`llvm/build-2d26/bin` |

通过 `BUDDY_PYTHON` 指定 Python 3.11；不要把 3.11 的 `.so` 塞进 3.12。
从仓库根目录准备官方 checkpoint 与 tokenizer 资源：

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
make -C "$MODEL" checkpoint BUDDY_PYTHON=/path/to/python3.11
make -C "$MODEL" resources BUDDY_PYTHON=/path/to/python3.11
```

官方 revision 为 `c1899de289a04d12100db370d81485cdf75e47ca`。
`model.safetensors` 为 1,503,300,328 字节，SHA256：
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
原 checkpoint 的 embedding 与 lm_head BF16 数据逐 bit 相等，部署只保留共享矩阵一份。
历史 `references/` 权重不是正常构建的隐式输入。

## 编译与执行入口

按 [REVIEW.md 的“重现新镜像”](REVIEW.md) 顺序执行完整命令。各层数必须使用同配置
的 import/layout、replacement report、prefill/decode IR 和 adapter，不能混用旧产物。

```text
import_model.py + weight_layout.py
  → 真实图、参数顺序和 checkpoint offset
triton_call_replace.py --w8a8 --attention
  → 结构匹配、external call、adapter、coverage
lower_model_nr.py (prefill / decode)
  → RISC-V LLVM IR / actual entry ABI
Triton → triton-riscv → MLIR → Buddy lowering → nr/kernel.o
  → build_model_lib.sh → libqwen_triton.a
build_nr_w8a8_segment.py
  → 权重段、scale、section/address/size/hash
build_nr_w8a8_image.py
  → graph object + adapter + Triton archive + 公共 NR → ELF/bin/map/audit
prepare_model_run.py
  → 按实际 ELF symbol 验证地址/边界，64-byte padding，DDR plan
run_model.sh → 公共 fpga_run.sh
  → 上传、分段加载、回读、运行、UART 记录
```

image builder 在实际 imported ABI 和 LLVM 类型上核验 shape/dtype/rank、层数、cache
容量、参数/结果描述符；错误直接失败。prefill 的 lm_head 已按共享权重语义改成仅计算
最后位置，输出 `[1,1,151936]`；不再复制整块转置权重。

静态库仅含 Triton kernel 和必要 ABI adapter，不含测试 main、测试数据、模型 oracle
或 NR runtime。`model-lib/evidence/`、`archive.json`、symbols、模型 link map 与 image
输入 hash 一起证明实际调用链。使用已验证的 AME int8 linear 和 RVV 辅助算子。

prepared run 目录的自动运行入口（替换 `<prepared-run>`）：

```bash
"$MODEL/tools/run_model.sh" <prepared-run> --fpga=5 \
  --capture-seconds=900 --startup-timeout=900
```

`fpga_run.sh` 支持 `--layout-plan` 多段上传和恢复已有会话。断线后复用已存在 run，
不要额外启动 minicom 或第二个 worker。服务器生成内容只允许在
`/home/hjuser/Desktop/fpga-tester-ISCAS` 内，不写软链接指向的外部目录。

## 数值验收与量化契约

```text
权重：per-output-channel symmetric int8 [-127,127]
scale=max(abs(row))/127，全零行 scale=1，zero-point=0
激活：per-token dynamic symmetric int8，使用同一舍入规则
舍入：FP32 除法 → 按符号加 ±0.5 → 截断 → 饱和
累加：int32，调用前清零 accumulator；不能误用累加语义为覆盖
反量化：(acc_f32 * activation_scale) * weight_scale，顺序固定
embedding/lm_head：共享同一量化矩阵及 per-row scale
norm/RoPE/attention/softmax/SiLU/residual/KV：FP32
```

独立 `quant_model_reference.py` 提供 `triton-host` 与 `nr-fpga` 算术 profile；后者
对应目标编译的融合与公共 NR math。跨 profile 的舍入可影响后续 int8 决策，不能混淆
为量化本身的误差。分别报告“图/FPGA vs 同 profile 独立量化参考”和“量化参考 vs FP32”。
“token 相同”“cosine 接近 1”或“比参考更近 FP32”都不能代替实现正确性验收。

自动模型镜像添加：

```text
--reference-arrays <quant-nr-order/arrays.npz>
--reference-metadata <quant-nr-order/quant-reference.json>
```

构建时核对 prompt、layers、capacity、profile、每步位置和 token trajectory。
参考 blob 是 ELF 中独立只读验证数据，只在图算完后比较结果；不选择 token，不更新 KV，
不放入 Triton archive。每次比较全词表末位 logits 与所有有效 K/V，UART 输出 count、
max/mean abs error；非有限值或超阈值即失败。

复核实际完成的 1 层 run：

```bash
python3 "$MODEL/tools/check_board_trace.py" \
  --uart "$MODEL/validation/board/review/model-1l-full/uart.raw.log" \
  --host-graph "$MODEL/build/review-1l/host-trace/arrays.npz" \
  --quant-reference "$MODEL/build/review-1l/quant-nr-order/arrays.npz" \
  --embedded-reference "$MODEL/validation/board/review/model-1l-full/numeric-reference.json" \
  --output "$MODEL/build/review-1l/rechecked-numeric-verification.json"
```

期望 exit=0、`FULL_LOGITS_KV_PASS`。该旧 oracle manifest 缺少后来增加的 dimensions
字段，保留原样并标 `metadata_verified=false`；独立归档证据另外核对了 NPZ→blob、
实际 ELF 内 blob、prompt/trajectory 及上传段 hash。prefill stage 的 `position=0`
表示图起点，tensor comparison 的 `position=15` 表示最后输出；decode 为 16–23。

这次 FPGA 全量比对没有覆盖 hidden state、全部 prefill 位置 logits 或 28 层。
与 compiled-host 的另一条比较仍是 9 个 logits 和每层一个 K/V 坐标的抽样。

## ABI、生命周期与内存

图中返回 tensor 的外部调用经 MLIR RAW bridge 到 `_mlir_ciface_*`。第一个指针是
**结果描述符存储槽**，callee 必须填入合法 descriptor，不能假定它已指向输出 buffer。
静态输出按 graph + call site 分配稳定 symbol，防止仍存活的同形状结果互相覆盖。
显式目标 memref 的 kernel 使用已规划输出，adapter 只转换 offset/stride/grid/rank。
参见 `validation/external-calls/return-descriptor-contract.md` 和实际生成 IR。

多层结果结构按真实类型生成：28 层结果共 6120 字节，不能使用一层的 288 字节结构。
每层 K/V 有独立 offset；每次图调用结束先保留所有返回 cache，再 reset scoped heap。
`cache_position` 是运行时数据，影响 RoPE、mask、有效 KV 长度和写入槽位。

最新 28 层非交互 ELF 静态计划（以 `validation/review-memory-28l.json` 为准）：

| HIGH 常驻部分 | 字节 |
| --- | ---: |
| int8 权重 + scales + 保留 FP32 | 598,230,784 |
| K/V cache（28×2×8×512×128×4） | 117,440,512 |
| prefill workspace | 218,800,512 |
| decode workspace | 124,927,872 |
| input IDs / position / 对齐 | 192 |
| **HIGH 合计** | **1,059,399,872（1,010.32 MiB）** |

LOW 静态区和 stack 为 19,527,680 字节，剩余 scoped heap 785,778,688 字节。
IR 中常量 malloc site 总和约 prefill 404 MB / decode 356 MB；这是静态 site 总和，
不是动态峰值实测。交互还要加入约 4.98 MiB tokenizer blob、表和 buffer，必须检查
最终交互 ELF。旧 737.99/764.14 MiB 只是预留估计，不能替代实际 ELF 计划。

NR linker 的 LOW `[0x80000000,0xb0000000)` 与 HIGH `[0xb8000000,0x100000000)`
是部署限制；平台 loader plan 要求 `ddr_size=0x400000000`（16 GiB），不代表容量已实测。
程序/权重段均按真实 ELF symbol 验证，`.bss` 清零与预载 NOLOAD 权重分离。

## tokenizer 和文本交互

裸机 encoder、decoder、template 无文件系统和动态分配；按锁定官方 tokenizer 的
added-token、NFC、Unicode regex 和 byte BPE 语义处理。细节和重现命令见
[text/README.md](text/README.md)。旧 `616/616`、`8/8 template`、`36/36 path` 是主机
历史验证；新板上证据是 `tokenizer-probe/` 的 **8 个完整文本 fixture**。

交互镜像用 `--interactive --tokenizer-blob <blob>` 构建并把 blob 加入 prepared plan。
它经 UART 读行、板上 template/tokenize、图 prefill/decode、argmax、增量文本解码。
变长 prompt 可逐 token 调用 decode 图；固定 16+8 数值验收仍使用一次 prefill。
当前为单轮会话（每次重置 cache），context=512，默认最多生成24 token；拒绝超长输入，
支持 CR/LF、退格、`/quit`、Ctrl-D，EOS 或 context 上界停止。

RA 通过公共 `nr_puts`/`nr_write` 输出，NH 实时排空 TX ring；**不能从 RA 直接轮询 UART**。
主机 `--interactive` 经唯一持有 UART 的 worker 转发 stdin，SSH 重连使用请求 UUID
避免重复投递。host write/ack 只证明 OS 接受字节，不证明 FPGA 接收。
当前 FPGA5 RX 根因尚未确认，未宣称 Stage E 通过。

## 剩余工作及历史证据

1. 使用修复后的图、adapter、runtime 和独立参考完成 4/28 层板测；真实 28 层 16+8
   必须比较完整末位 logits、逐步有效 KV、关键 hidden state 并记录 cycles。
2. 新的动态 attention 优化先单 kernel Stage A，再集成模型验证；未验收变体不能替代
   已有模型结果。RoPE 当前由 Buddy 图 lowering，仍是明确未替换项。
3. 定位 FPGA5 RX，再验证真实 UART 文本输入→tokenizer→28层生成→增量文本输出。
4. 补足各主要 kernel 的调用数/cycles、内存复制/同步开销和实际 heap 峰值。

旧板测/主机 JSON 保留在 `validation/`；索引的 `historical_stages` 明确标注原始范围和
失效结论。旧 28 层长跑没有完成，也不能凭控制进程 CPU、链接通过或一层正确排除崩溃。
本次不把任何单 kernel、链接审计、可读文本或主机结果提升为 28 层 FPGA 端到端验收。
