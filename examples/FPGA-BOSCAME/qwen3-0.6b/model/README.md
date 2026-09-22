# Qwen3-0.6B：Buddy 计算图 + Triton 静态库 → NR FPGA

默认部署为 **完整 28 层、batch=1、最大序列长度 128、完整词表 151936、W8A8 linear**。
最大长度包含 prompt 和生成 token。固定验收使用原始文本 `What is France?`，在 FPGA
上应用 chat template、编码为 16 token，执行一次 prefill，保留 KV 后连续 decode 8 步
（position 16…23），由 FPGA 选择 token 并增量解码输出文本。

2026-09-18，启用共享激活动态量化及 RVV 量化的 FPGA5
`run-c9c51a235a154732` 已通过严格数值验收：

| 检查 | 结果 |
| --- | --- |
| 完整末位 logits、全部有效 K/V | 9 个阶段共 27 项通过；全部 KV 零误差；首次 decode logits 最大绝对误差 9.536743e-7，其余 logits 零误差 |
| 板上文本处理 | tokenizer、token 选择及增量文本解码通过 |
| 装载与退出 | 三段 DDR 回读一致，`RA returned: PASS`，runner `status: OK` |
| 严格归档 | `MODEL_RUN_NUMERIC_PASS`、`FULL_LOGITS_KV_PASS`、`FIXED_TEXT_PASS` |

证据见 [verification.json](validation/board/quant-opt/model-28l-cap128/verification.json)，
时间见 [performance.json](validation/board/quant-opt/performance.json)。
Prefill 图计算 **247.92 秒**；decode 平均图计算 **68.00 秒/token**，计入准备、选择和
KV 保留后 **71.43 秒/token**；整个 launch 含初始化、验证和输出约 **14.13 分钟**。
时间按配置时钟 14.7456 MHz 换算，不含主机上传和 DDR 装载；该验收镜像没有逐 kernel profiler。

2026-09-17 的原验收基线 `run-2c2dd9c210c1492a` 保留在
[原归档](validation/board/ame-v05/model-28l-cap128/verification.json)。其 prefill 为
382.35 秒，decode 图计算为 76.59 秒/token，含准备/选择/KV 为 80.01 秒/token。
本次对应减少 **35.16%、11.21%、10.72%**，token 轨迹及数值误差保持一致。

本次预测 token（含 prefill 的首次预测）为：

```text
49000,374,264,3146,7407,304,4787,5159,11
France is a country located in North America,
```

这段文本与同 checkpoint 的 FP32 固定用例 token 轨迹一致。它验证执行链数值正确，
不表示生成内容事实正确。当前使用预置文本，**UART RX 交互未完成**；主机不参与板上
tokenizer、推理、KV 更新、token 选择或文本解码。

## 快速复跑已有镜像

在仓库根目录执行。下列路径对应本次已验收的本地产物；`build/`、权重和二进制不提交
Git，新 checkout 需要先按后面的步骤重建。

```bash
REPO="$(git rev-parse --show-toplevel)"
MODEL="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/model"
OUT="$MODEL/build/quant-opt/shared"
"$MODEL/tools/run_model.sh" "$OUT/prepared" \
  --fpga=5 --capture-seconds=2400 --startup-timeout=900
```

记录脚本打印的 `run-ID`。脚本上传 boot image、量化权重和 tokenizer 三段，校验回读，
启动模型并收集 UART。一次任务只能有一个 UART owner；运行时不要再开 minicom。

SSH 断开不表示 FPGA 停止。使用实际 run ID 恢复原 worker，不重新上传或 reset：

```bash
RUN_ID=run-c9c51a235a154732  # 换成本次任务打印的 ID
"$REPO/examples/FPGA-BOSCAME/fpga_run.sh" "$OUT/prepared/image.bin" \
  --fpga=5 --resume-run="$RUN_ID"
```

恢复必须直接调用公共 `fpga_run.sh`，不要调用会附加上传段的 `run_model.sh`。
之前的验收已验证在 SSH 中断后恢复原 worker、取回完整结果，无需重启模型。

## 从源码重建：环境与资源

如果没有任何构建目录或已编译工具链，请使用
[从零构建 none 整模型](../README.md#构建-none-整模型二进制)：其中包含工具链、
全新 Python 环境、显式 kernel 清单和全部资源的生成，不依赖下面的历史构建环境。

下面各段按顺序在同一个 Bash 会话运行。使用新的输出目录，保留已验收产物。
需先具备本仓库已编译的 Buddy、LLVM、Python bindings 和 triton-riscv；环境设置见
[公共 Triton 说明](../../common/triton/README.md)。`examples/BuddyQwen3` 不需修改。

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
MODEL="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/model"
OUT="$MODEL/build/quant-opt/model-28l-cap128-reproduce"
# 按机器环境设置；Buddy Python 必须与其 MLIR bindings 的 Python ABI 一致。
export BUDDY_PYTHON="${BUDDY_PYTHON:-/home/chh/venvs/qwen3fpga-py311/bin/python}"
export TRITON_PYTHON="${TRITON_PYTHON:-/home/chh/miniconda3/envs/boscame/bin/python}"
export PYTHONPATH="$REPO/build-python/python_packages"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
LINKER="$(command -v ld.lld-20)"
source "$REPO/examples/FPGA-BOSCAME/common/triton/triton-env.sh"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo '请改用新的 OUT'; exit 1; }
mkdir -p "$OUT"
make -C "$MODEL" assets checkpoint resources PYTHON="$BUDDY_PYTHON"
PROMPT=151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271
COMMON=(--assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint"
        --layers 28 --prefill-len 16 --max-cache-len 128)
```

Buddy 使用 Python 3.11 bindings，当前 `boscame` 的 Triton 使用 Python 3.12，两者分开
调用。链接器要求 LLD >= 20。官方资源锁定 revision
`c1899de289a04d12100db370d81485cdf75e47ca`，checkpoint SHA256 为
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
构建不依赖 `references/` 中的历史权重。

## 编译 46 个 Triton kernel

从本次归档读取精确 case 清单，重新执行 Triton → triton-riscv → linalg → Buddy
lowering → object。清单仅选择算子，不复用归档里的旧目标文件。
11 个 INT8 linear 使用 AME N64、直接 GPR 编码及已验证的 fence 合并；7 个反量化算子
和 6 个动态量化算子使用 RVV。不同配置用独立目录，最终 overlay 仅用于读取。
量化仍使用 FP32 除法及原有舍入规则，优化细节和单算子 A/B 复现见
[QUANTIZE.md](optimization/QUANTIZE.md)。

```bash
CASES="$OUT/cases"
KERNELS="$OUT/kernels"
OVERLAY="$OUT/kernel-overlay"
"$BUDDY_PYTHON" -B "$MODEL/tools/model_kernel_cases.py" \
  --capacity 128 --output "$CASES/model"
"$BUDDY_PYTHON" -B "$MODEL/tools/attention_position_cases.py" \
  --capacity 128 --output "$CASES/position"
"$BUDDY_PYTHON" -B "$MODEL/tools/attention_position_cases.py" \
  --capacity 128 --native-key --output "$CASES/native"
export QWEN_CASE_ROOTS="$CASES/model:$CASES/position:$CASES/native"
EVIDENCE="$MODEL/validation/board/ame-v05/model-28l-cap128/build/model-lib/archive.json"
"$BUDDY_PYTHON" - "$EVIDENCE" "$OUT" <<'PY'
import json, pathlib, sys
records = json.load(open(sys.argv[1]))['cases']
groups = {'ame': [], 'rvv': [], 'base': []}
for r in records:
    family = r['frontend']['family']
    group = 'ame' if family == 'matmul_i8' else 'rvv' if family == 'dequantization' else 'base'
    groups[group].append(r['case'])
assert {k: len(v) for k, v in groups.items()} == {'ame': 11, 'rvv': 7, 'base': 28}
for group, names in groups.items():
    (pathlib.Path(sys.argv[2]) / (group + '-cases.txt')).write_text('\n'.join(sorted(names)) + '\n')
PY
export QWEN_TRITON_QUANT=rvv
for group in base ame rvv; do
  args=()
  while IFS= read -r name; do args+=(--case "$name"); done < "$OUT/$group-cases.txt"
  export QWEN_TRITON_BUILD_ROOT="$KERNELS/$group"
  unset QWEN_TRITON_AME_N QWEN_TRITON_DEQUANT
  export QWEN_MAKE_FLAGS="RISCV_LD=$LINKER AME_GPR_MODE=fixed NR_COALESCE_FENCES=0"
  if [[ "$group" == ame ]]; then
    export QWEN_TRITON_AME_N=64
    export QWEN_MAKE_FLAGS="RISCV_LD=$LINKER AME_GPR_MODE=direct NR_COALESCE_FENCES=1"
  elif [[ "$group" == rvv ]]; then
    export QWEN_TRITON_DEQUANT=rvv
  fi
  "$TRITON_PYTHON" -B "$MODEL/../triton/build.py" "${args[@]}" --jobs 2
  "$TRITON_PYTHON" -B "$MODEL/../triton/build.py" "${args[@]}" --jobs 2 --host
done
unset QWEN_TRITON_BUILD_ROOT QWEN_TRITON_AME_N QWEN_TRITON_DEQUANT QWEN_TRITON_QUANT QWEN_MAKE_FLAGS
"$BUDDY_PYTHON" -B "$MODEL/tools/kernel_build_overlay.py" \
  --source "$KERNELS/base" --source "$KERNELS/ame" --source "$KERNELS/rvv" \
  --output "$OVERLAY"
```

`--host` 是独立主机验证产物，不执行 FPGA AME 指令。不要把编译器输出指向 overlay
中的软链接。原 AME/反量化证据位于 `validation/board/ame-v05/`，新增量化及模型
证据位于 `validation/board/quant-opt/`。

## 导入真实模型图、替换与 lowering

```bash
"$BUDDY_PYTHON" -B "$MODEL/tools/import_model.py" "${COMMON[@]}" \
  --output "$OUT/import" --fuse none --save-mlir --init-cache
"$BUDDY_PYTHON" -B "$MODEL/tools/weight_layout.py" \
  --mlir "$OUT/import/forward_decode.mlir" --params "$OUT/import/params.json" \
  --checkpoint "$MODEL/assets/checkpoint/model.safetensors" \
  --config "$MODEL/assets/official/config.json" --output "$OUT/import/weight-layout.json"
"$BUDDY_PYTHON" -B "$MODEL/tools/triton_call_replace.py" "${COMMON[@]}" \
  --triton-build "$OVERLAY" --output "$OUT/replacement" \
  --attention --attention-position --attention-native-key --w8a8 --share-activation-quantization
for kind in prefill decode; do
  "$BUDDY_PYTHON" -B "$MODEL/tools/lower_model_nr.py" "${COMMON[@]}" \
    --triton-build "$OVERLAY" --output "$OUT/nr-$kind" --kind "$kind" \
    --attention --attention-position --attention-native-key --w8a8 --share-activation-quantization
done
TRITON_BUILD="$OVERLAY" OUT="$OUT/model-lib" \
  ADAPTERS="$OUT/replacement/qwen_triton_adapters.c" LD_LLD="$LINKER" \
  bash "$MODEL/tools/build_model_lib.sh"
TRITON_BUILD="$OVERLAY" OUT="$OUT/host-bridge" \
  ADAPTERS="$OUT/replacement/qwen_triton_adapters.c" \
  bash "$MODEL/tools/build_host_bridge.sh"
```

每图应有 873 个外部 kernel 调用，未覆盖的大计算为 0。Q/K/V 及 gate/up 共享
已经证明相同的 RMSNorm 输入量化，每图 Quantize 从 197 次降至 113 次；
`activation_quantization` 报告应为 `created=113, reused=84`。图 ABI、replacement report、
adapter、KV 容量和权重 layout 必须来自同一次配置。静态库包含 46 个 kernel 的
92 个 kernel/adapter 对象；不包含测试 main、测试数据或公共 NR runtime。
`model-lib/archive.json`、`evidence/`、symbol/link report 和最终 ELF map 保留调用链证据。

## 独立参考与主机编译图验证

FP32、量化参考和主机编译图只用于验证，不参与 FPGA 推理。板上使用 `nr-fpga`
算术 profile；主机编译图对照 `triton-host`，避免将目标端融合/舍入差异混为量化误差。

```bash
"$BUDDY_PYTHON" -B "$MODEL/tools/host_reference.py" \
  --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
  --layers 28 --max-cache-len 128 --prompt-ids "$PROMPT" --decode-steps 8 \
  --output "$OUT/fp32-reference"
for profile in nr-fpga triton-host; do
  "$BUDDY_PYTHON" -B "$MODEL/tools/quant_model_reference.py" \
    --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
    --layout "$OUT/import/weight-layout.json" --layers 28 --max-cache-len 128 \
    --prompt-ids "$PROMPT" --decode-steps 8 --arithmetic-profile "$profile" \
    --fp32-reference "$OUT/fp32-reference" --output "$OUT/quant-$profile"
done
"$BUDDY_PYTHON" -B "$MODEL/tools/run_graph_host.py" \
  --assets "$MODEL/assets/official" --checkpoint "$MODEL/assets/checkpoint" \
  --layers 28 --max-cache-len 128 --output "$OUT/host-run" \
  --layout "$OUT/import/weight-layout.json" --triton-build "$OVERLAY" \
  --external-lib "$OUT/host-bridge/libqwen_triton_host.so" \
  --replace --attention --attention-position --attention-native-key --w8a8 --share-activation-quantization \
  --decode-steps 8 --capture-intermediates --prompt-ids "$PROMPT" \
  --quant-reference-dir "$OUT/quant-triton-host"
```

验收记录中的主机编译图与同 profile 参考，27 项完整 logits/KV 比较均零误差。
权重采用 per-output-channel symmetric INT8，激活采用 per-token dynamic symmetric INT8，
zero-point=0；INT32 accumulator 清零后累加，再按 `(acc_f32 * activation_scale) * weight_scale`
反量化。舍入为 FP32 除法、按符号加 ±0.5、截断并饱和到 [-127,127]；全零行 scale=1。
embedding/lm_head 共享量化矩阵与 scale，norm、RoPE、attention、SiLU、residual、KV 使用 FP32。

## 权重、ELF 与三段 DDR 镜像

下面显式使用原始 none 基线的启动/日志配置：无额外启动同步，图结束仍做
AME resync，console 为 4 MiB append-only 并实时 drain，不启用额外诊断。
仅写 `--ame-startup=none` 而保留默认 ring console，并不等于该基线配置。
不具备已有输入时，使用[从零构建步骤](../README.md#构建-none-整模型二进制)。

```bash
"$BUDDY_PYTHON" -B "$MODEL/tools/build_nr_w8a8_segment.py" \
  --report "$OUT/replacement/triton-call-replacement.json" \
  --layout "$OUT/import/weight-layout.json" \
  --checkpoint "$MODEL/assets/checkpoint/model.safetensors" \
  --config "$MODEL/assets/official/config.json" --output "$OUT/weights"
"$BUDDY_PYTHON" -B "$MODEL/tools/build_nr_w8a8_image.py" \
  --repo-root "$REPO" --linker "$LINKER" \
  --report "$OUT/replacement/triton-call-replacement.json" \
  --segment "$OUT/weights/w8a8-segment.json" \
  --graph-ir "$OUT/nr-prefill/forward_prefill.ll" \
  --decode-ir "$OUT/nr-decode/forward_decode.ll" \
  --archive "$OUT/model-lib/libqwen_triton.a" \
  --adapters "$OUT/replacement/qwen_triton_adapters.c" \
  --output "$OUT/image" --layers 28 --cache-len 128 --prefill-len 16 --decode-steps 8 \
  --prompt-ids "$PROMPT" --prompt-text 'What is France?' \
  --tokenizer-blob "$MODEL/build/tokenizer.bin" \
  --reference-arrays "$OUT/quant-nr-fpga/arrays.npz" \
  --reference-metadata "$OUT/quant-nr-fpga/quant-reference.json" \
  --ame-startup=none --graph-sync=ame-resync --ame-cache-sync=none \
  --console-mode=append-only --console-capacity=4194304 --console-drain=live
"$BUDDY_PYTHON" -B "$MODEL/tools/prepare_model_run.py" \
  --image "$OUT/image/qwen_model.bin" --elf "$OUT/image/qwen_model.elf" \
  --weights "$OUT/weights/weights-w8a8.bin" \
  --weight-manifest "$OUT/weights/w8a8-segment.json" \
  --tokenizer "$MODEL/build/tokenizer.bin" --output "$OUT/prepared"
```

以上只在本机构建与准备 DDR 文件，不会上板。需要运行新镜像时，再单独执行：

```bash
"$MODEL/tools/run_model.sh" "$OUT/prepared" \
  --fpga=5 --remote-dir=/home/hjuser/Desktop/fpga-tester-ISCAS \
  --capture-seconds=1800 --startup-timeout=900
```

此命令不启用 kernel profiler。参考数组仅在模型计算后做比较，不用于生成 token 或更新 KV。
公共 `common/nr` 负责启动、通信和同步，ELF audit 应无未定义符号。
当前源码重编不保证与历史 `production-control` ELF 逐字节一致；
新镜像必须重新验收，不能沿用原始 none 的 PASS 记录。

本次权重段 598,230,784 字节，tokenizer 5,222,976 字节；HIGH 持久区总计
705,093,888 字节，范围约 `0xb8000000..0xe206e100`，包含权重、KV、tokenizer 和 workspace。
LOW heap 可用 774,796,928 字节；实测 prefill/decode scratch 峰值分别为
134,597,013 / 91,611,925 字节。实际装载地址、大小与 hash 以当前生成的 manifest 和
`prepared/ddr-load.plan` 为准；startup 清零范围不得覆盖外部装载资源。

## 结果归档与校验

将 `RUN_ID` 换为本次实际完成的任务，`ARCHIVE` 选一个不存在的新目录：

```bash
RUN_ID=run-ACTUAL_ID
ARCHIVE="$MODEL/validation/board/quant-opt/model-28l-cap128-reproduce"
"$BUDDY_PYTHON" -B "$MODEL/tools/archive_model_run.py" \
  --run "$REPO/examples/FPGA-BOSCAME/build/fpga-runs/$RUN_ID" \
  --build "$OUT" --image-dir "$OUT/image" --prepared-dir "$OUT/prepared" \
  --host-graph "$OUT/host-run/arrays.npz" \
  --quant-reference "$OUT/quant-nr-fpga/arrays.npz" \
  --reference-metadata "$OUT/quant-nr-fpga/quant-reference.json" \
  --layers 28 --prefill 16 --steps 8 --assets "$MODEL/assets/official" \
  --output "$ARCHIVE"
```

要求 `MODEL_RUN_NUMERIC_PASS`，并核对 `numeric-verification.json`、
`fixed-text-verification.json`、`run/uart.raw.log`、`run/result.json`。
归档会校验镜像/输入 hash、DDR 回读、最终退出、完整 logits/KV 和板上文本处理；
只看到文本或 `RA returned: PASS` 不足以替代数值验收。重新编译的文件 hash 可能变化，
必须用新镜像对应的 build 和 run 验证，不能沿用旧 PASS。

## FPGA 服务器手动操作

如果自动任务仍在运行，只查看其日志，不再次启动模型：

```bash
ssh fpga
cd /home/hjuser/Desktop/fpga-tester-ISCAS
RUN_ID=run-ACTUAL_ID
tail -n 100 -f "fpga-runs/$RUN_ID/uart.raw.log"
```

确认原 worker 已退出、锁已释放后，才可用已经上传的同一组三段文件手动重跑。
服务器上的两个终端都必须在 `/home/hjuser/Desktop/fpga-tester-ISCAS` 内：

终端 A（唯一串口读取进程）：

```bash
cd /home/hjuser/Desktop/fpga-tester-ISCAS
RUN_ID=run-ACTUAL_ID
minicom -b 115200 -D /dev/FPGA5 -C "fpga-runs/$RUN_ID/manual-minicom.log"
```

终端 B：

```bash
cd /home/hjuser/Desktop/fpga-tester-ISCAS
RUN_ID=run-ACTUAL_ID
UV_RUN_READBACK=1 UV_RUN_SYS_CLK_HZ=14745600 TMPDIR="$PWD/fpga-runs/$RUN_ID" \
  make uv_run5 layout="fpga-runs/$RUN_ID/ddr-load.plan"
```

必须用 `layout=` 装载 boot image、权重及 tokenizer，不能只用 `test=image.bin`。
手动运行日志不属于原自动 worker 的验收，不能将原 `result.json` 当成此次成功记录；
需要可重复严格归档时使用前述自动运行入口。服务器上传、日志及操作限定在上述目录，
不修改其中软链接实际指向的外部目录。
