# 2026-09-17 工程复核与继续验证

**完整 28 层 FPGA 16+8 和板上文本交互尚未验收。** 旧报告的部分 PASS 超出了原始证据；
旧 JSON 的原始数值保留，新增 `review_2026_09_17` 标明失效范围。
本次未改 git index，也未修改 `examples/BuddyQwen3`。

## 直接修复的问题

| 问题 | 原后果 | 修复与验证 |
| --- | --- | --- |
| 返回结构固定一层 | 28 层实际返回 6120 字节，C 只提供 288 字节，栈越界 | 从实际 LLVM 核验每个参数和返回 descriptor rank；按层生成结构；1/4/28 层 ASan/UBSan 调用测试 |
| 所有层的 K/V 指向同一 base | 多层 KV 互相覆盖 | 每层独立 offset；保留全部层的返回 cache |
| “full 1L” 的 report 缺 10 个 workspace 参数 | 59 个实参接到 69 个 descriptor 的函数 | report/adapter/prefill/decode 同配置重建，image builder 遇 ABI 不一致立即失败 |
| runtime `free` 不回收，decode 持续 bump allocate | 多步或多轮最终耗尽 heap | 一次 graph call 一个 scratch scope；保存所有 cache 后 reset；耗尽输出诊断并失败 |
| 多处调用共享一个静态输出 | 同 shape 但仍存活的结果可能相互覆盖 | 返回 tensor 的 wrapper 按 graph+call site 分配稳定 symbol/输出存储 |
| 交互重放 prompt 最后一个 token | 首个生成 token 和 cache position 错位 | 使用最后一次 prompt forward 已产生的预测；测试精确核对输入 token/position 轨迹 |
| 交互输入静默截断、无 context 检查 | 输入与推理内容不一致，可能 KV 越界 | 显式拒绝超长输入、限制 context，支持 CR/LF、退格、`/quit`、Ctrl-D |
| RX ring 初始化/缓存所有权错误 | 旧 head/tail 或缓存行覆盖 | NH/RA 分离 cache line，NH 初始化，单次读取 UART RBR |
| TX 为累计 64 KiB 缓冲 | 多轮交互迟早丢输出 | 公共 runtime 改为带消费确认的 ring，板测 67584 字节逐字节一致 |
| tokenizer NFC/UTF-8/ctypes 错误 | Unicode 输入错分词、越界或静默截断 | 直接提取锁定 tokenizer 的 Unicode 规则，差分与 sanitizer 验证，见 `text/README.md` |
| 裸机未填 attention/KV position workspace | 所有 token 重复写同一 cache slot，因而输出错误 | 每次调用按实际位置生成 int32 `position + arange(S)`，与 RoPE position 一致 |
| 主机 KV 验证读 prefill 快照 | decode cache 证据无效 | 保存各 step 的实际返回 cache 和最终有效区域 |
| 替换语义检查不足 | 非目标 transpose/norm/attention/index 操作也可能匹配 | 增加语义否定用例、实际模型 IR ABI、未覆盖大计算报告 |

image builder 现在只负责 ABI、持久内存和 session 控制。decoder 执行仍来自 Buddy 图；
算术仍经 Triton → MLIR → Buddy 静态库。没有用 C 重写模型计算。

## 证据边界

- 旧 `616/616` 是主机上的 C encoder 对官方 tokenizer；不是 FPGA 测试。
- `0.994366 graph-vs-quantized-reference cosine` 不能以“更接近 FP32”为由判实现通过。
  新主机测试同时保存 graph vs FP32 和 graph vs 独立量化参考，超过误差阈值会明确失败。
- `uv_file_process` / 控制进程使用 CPU，不是 RA PC/cycle 进度证据。
- 平台 `u2_set_partition.tcl` 配置了 TX 和 RX 引脚。FPGA5 当前 RX 探针未收到字节；
  根因尚未确认，不能宣称整个 NR 平台没有 RX。
- DDR canary 只证实 19 个抽样地址的 1216 字节读写/不互相 alias；不等于全容量扫测。
- `RA returned: PASS` 只表示固件返回成功。必须再执行数值比对，不能代替数值验收。
- 当前 UART trace 是 argmax、9 个选定 logits、每层 head0/dim0 的 KV 样本与 cycles；
  未启用 numeric oracle 时，`check_board_trace.py` 明确把它标成 **sampled**。
  启用下述验证固件后另有板上完整 logits/有效 KV 比对，不能混淆两种证据。

## 重现新镜像

以下命令从仓库根目录执行。Python 3.11 用于 Buddy graph；Triton 构建使用已安装
triton-riscv 的 boscame 环境。两者不能混用 MLIR bindings 的 Python ABI。
`CHECKPOINT` 必须指向官方真实权重目录，不能指向历史 references 数据。

```bash
REPO="$PWD"
MODEL="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/model"
# 先激活含 torch/transformers 和 Buddy Python 3.11 bindings 的环境。
BUDDY_PYTHON="$(command -v python)"
export PYTHONPATH="$REPO/build-python/python_packages"
ASSETS="$MODEL/assets/official"
CHECKPOINT="$MODEL/assets/checkpoint"
LAYERS=1                         # 正式验收用 28；1/4 是定位用缩小层数
OUT="$MODEL/build/review-${LAYERS}l"
TRITON_BUILD="$MODEL/../triton/build"
mkdir -p "$OUT"

# 原始真实图与权重 offset；每个层数都必须有相应的 layout。
"$BUDDY_PYTHON" -B "$MODEL/tools/import_model.py" \
  --assets "$ASSETS" --checkpoint "$CHECKPOINT" --output "$OUT/import" \
  --layers "$LAYERS" --prefill-len 16 --max-cache-len 512 --fuse none \
  --init-cache --save-mlir
"$BUDDY_PYTHON" -B "$MODEL/tools/weight_layout.py" \
  --mlir "$OUT/import/forward_decode.mlir" --params "$OUT/import/params.json" \
  --checkpoint "$CHECKPOINT/model.safetensors" --config "$ASSETS/config.json" \
  --output "$OUT/weight-layout.json"

"$BUDDY_PYTHON" -B "$MODEL/tools/triton_call_replace.py" \
  --assets "$ASSETS" --checkpoint "$CHECKPOINT" --triton-build "$TRITON_BUILD" \
  --output "$OUT/replacement" --layers "$LAYERS" --prefill-len 16 \
  --max-cache-len 512 --w8a8 --attention
for kind in prefill decode; do
  "$BUDDY_PYTHON" -B "$MODEL/tools/lower_model_nr.py" \
    --assets "$ASSETS" --checkpoint "$CHECKPOINT" --triton-build "$TRITON_BUILD" \
    --output "$OUT/nr-$kind" --layers "$LAYERS" --prefill-len 16 \
    --max-cache-len 512 --kind "$kind" --w8a8 --attention
done
ADAPTERS="$OUT/replacement/qwen_triton_adapters.c" OUT="$OUT/model-lib" \
  bash "$MODEL/tools/build_model_lib.sh"

# 权重段先于 ELF；无需伪造一个 image.bin 才能准备权重。
"$BUDDY_PYTHON" -B "$MODEL/tools/build_nr_w8a8_segment.py" \
  --report "$OUT/replacement/triton-call-replacement.json" \
  --layout "$OUT/weight-layout.json" --checkpoint "$CHECKPOINT/model.safetensors" \
  --config "$ASSETS/config.json" --output "$OUT/weights"
python3 -B "$MODEL/tools/build_nr_w8a8_image.py" --repo-root "$REPO" \
  --report "$OUT/replacement/triton-call-replacement.json" \
  --segment "$OUT/weights/w8a8-segment.json" \
  --graph-ir "$OUT/nr-prefill/forward_prefill.ll" \
  --decode-ir "$OUT/nr-decode/forward_decode.ll" --decode-steps 8 \
  --archive "$OUT/model-lib/libqwen_triton.a" \
  --adapters "$OUT/replacement/qwen_triton_adapters.c" \
  --output "$OUT/image" --layers "$LAYERS"
python3 -B "$MODEL/tools/prepare_model_run.py" \
  --image "$OUT/image/qwen_model.bin" --elf "$OUT/image/qwen_model.elf" \
  --weights "$OUT/weights/weights-w8a8.bin" --weight-manifest "$OUT/weights/w8a8-segment.json" \
  --output "$OUT/run"
"$MODEL/tools/run_model.sh" "$OUT/run" --fpga=5 --capture-seconds=900 --startup-timeout=900
```

`prepare_model_run.py` 从最终 ELF symbol 取得地址、检查段边界和 hash、进行 64-byte padding。
远端 plan 的 16 GiB DDR header 是平台 loader 要求；实际段额外受 NR linker 的
`[0x80000000,0xb0000000)` / `[0xb8000000,0x100000000)` 限制。两者含义不同。
上板时所有生成内容在服务器 `fpga-tester-ISCAS` 内，不能写外部软链接目标。

Triton kernel 源码重建复用公共入口：先生成 model case metadata，再在 boscame 中
source `common/triton/triton-env.sh`，设置 `QWEN_CASE_ROOTS=$MODEL/build/model-cases`，
运行 `python triton/build.py --case ...`。替换器只接受已有 frontend metadata 的特化；
静态库脚本检查对应 `nr/kernel.o`，不包含测试 main/runtime/data。
`build/model-lib/evidence/` 的 frontend/MLIR/object hash 是编译链证据。

## 自动数值记录

`tools/check_board_trace.py --uart <uart.raw.log> --host-graph <arrays.npz>
--quant-reference <arrays.npz> --layers <N> --output <report.json>` 比较两条参考，检查
完整 prefill+8 decode 记录和每层样本，任何缺失/非有限值/阈值超限都不判通过。
`compute_cycles` 仅包围 graph call 和 AME 完成 fence；
`total_with_uart_cycles` 含采样/复制/日志，不能当作纯模型计算时间。

文本交互镜像在 image build 命令增加 `--interactive --tokenizer-blob <tokenizer.bin>`；
prepare 命令也增加 `--tokenizer <tokenizer.bin>`。单轮状态每次新输入重置，context=512，
默认最多24个新 token；prompt 超界拒绝，generation 到 context 上界停止。
当前 FPGA5 RX 故障尚未定位，不能将生成了交互 ELF 说成真实交互通过。

基础本地回归：

```bash
python3 -B -m unittest discover -s "$MODEL/tests" -p 'test_*.py' -v
make -C "$MODEL" check BUDDY_PYTHON="$BUDDY_PYTHON"
```

## 完整数值自动验收固件

image builder 可加 `--reference-arrays <独立 quant-nr-order/arrays.npz>`，将验证用参考
logits/KV 作为独立只读 section 链入 ELF。此资源只用于**结果比对**，不参与 token 选择、
KV 更新或任何模型算术；kernel 静态库内没有这份数据。每次真实 graph call 返回后，板上
遍历全部 vocabulary logits 和当前有效 K/V，输出 max/mean abs error。默认阈值分别
`1e-3`/`1e-4`，可显式配置；不满足就返回失败，不能将相同 token 当作数值通过。

验证器增加 `--embedded-reference <image/numeric-reference.json>`，检查 NPZ hash 对应、
27 条完整 tensor 检查的 count、位置、阈值及固件完成状态。`FULL_LOGITS_KV_PASS` 只覆盖
所指定层数和这些 tensor，不自动覆盖完整28层或 hidden-state 分段验收。

量化主机参考有明确的算术 profile：`triton-host` 对应独立顺序 FP32/host libm，
`nr-fpga` 对应从目标代码核对的融合运算与公共 NR math。不同 reduction/softmax 算术
可能跨越 int8 的半整数舍入边界；不能把 profile 不匹配的误差解释成纯量化误差。

新的 image builder 同时读取 NPZ 同目录的 `quant-reference.json`（可通过
`--reference-metadata` 指定），核验 prompt IDs、层数、cache 容量、`nr-fpga` profile，
以及每个 decode 的位置、输入 token 和 logits argmax。参考数据与部署配置不一致时
构建失败。元数据及 NPZ 的 hash 都记录在 `numeric-reference.json`。

### 已完成的复核板测

`run-015b4fabf4014edc` 使用真实一层图、完整替换集、官方量化权重，完成一次 16-token
prefill 和连续 8 步 decode（位置 16..23）。每次调用的最后位置全部 151936 logits
和全部有效 K/V，共 27 项 tensor 检查，max/mean abs error 均为零；prefill argmax=33067，
后续八步均为11853。权重与程序段 DDR readback 均匹配，NR 返回 PASS。
原始日志与严格验收位于 `validation/board/review/model-1l-full/`。

该镜像先于 oracle 元数据字段扩展构建，原 manifest 原样保存，验收报告明确
`metadata_verified=false`。独立参考 NPZ hash 与实际镜像中的 oracle 一致；没有回填
旧镜像的构建记录。此测试尚未覆盖中间 hidden state、完整28层或外部 UART 输入。

### 可选 kernel 计时

image builder 的 `--profile-kernels` 用链接器 `--wrap` 包围 adapter 实际引用的 Triton
符号，记录每个 graph call 内的 kernel 次数与 cycles，在 graph 返回后输出 `[profile]`。
`kernel-profile.json` 保存原符号、完整指针 ABI、wrapper source hash 和链接参数。
wrapper 只转发原参数、计时并等待 AME 完成，不实现算子、不改变输出。

此模式增加每个 kernel 返回后的 fence 和计数开销，`compute_cycles` 也包含这些开销；
不能当作未插桩性能。kernel cycles 之和以外仍包括图内 materialization/copy、标量操作
和计数开销，不能把差值全部归为内存耗时。UART 打印发生在 graph 计时之外。
