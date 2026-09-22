# 完整模型的逐 kernel 计时

配置为完整 28 层、max sequence=128、16-token prefill＋8 步 decode，使用量化
共享及 RVV 量化优化。计时包装通过链接器 `--wrap` 调用原 Triton kernel，复用
已验收图、权重和静态库；不改变 kernel 计算。每图 873 次调用，46 个特化符号。

计时包括 kernel 的 descriptor/grid adapter 和调用末尾的 AME 完成同步；同一
kernel/shape 在所有层的时间累加。未开启逐调用 UART、phase probe 或中间结果
比较。报告保留每个 decode position 的计时，并提供 8 步均值。

`graph - kernel totals` 单独列出：包含图内 memcpy、view 物化、RoPE、residual、
其它标量工作、图 adapter、分配和计时包装开销。当前 profiler 无法进一步分开
这些项目，不能将这部分全部归因于访存。单次调用耗时列为均值，不是逐层测量。
时间按照配置的 14.7456 MHz 换算。

## 复现命令

在仓库根目录执行。BASE 指向已验收的量化优化构建，REF 提供独立数值参考。
选一个不存在的 OUT，避免覆盖旧证据。已有任务在运行时只恢复任务，不重复启动。

```bash
MODEL="$PWD/examples/FPGA-BOSCAME/qwen3-0.6b/model"
BASE="$MODEL/build/quant-opt/shared"
REF="$MODEL/build/ame-v05/model-28l-cap128"
OUT="$MODEL/build/quant-opt/profile-28l-cap128-reproduce"
BUDDY_PYTHON="${BUDDY_PYTHON:-/home/chh/venvs/qwen3fpga-py311/bin/python}"
"$BUDDY_PYTHON" -B "$MODEL/tools/build_nr_w8a8_image.py" \
  --repo-root "$PWD" --linker /usr/bin/ld.lld-20 \
  --report "$BASE/replacement/triton-call-replacement.json" \
  --segment "$BASE/weights/w8a8-segment.json" \
  --graph-ir "$BASE/nr-prefill/forward_prefill.ll" \
  --decode-ir "$BASE/nr-decode/forward_decode.ll" \
  --archive "$BASE/model-lib/libqwen_triton.a" \
  --adapters "$BASE/replacement/qwen_triton_adapters.c" \
  --output "$OUT/image" --layers 28 --cache-len 128 \
  --prefill-len 16 --decode-steps 8 --profile-kernels \
  --prompt-text 'What is France?' --tokenizer-blob "$MODEL/build/tokenizer.bin" \
  --reference-arrays "$REF/quant-nr-fpga/arrays.npz" \
  --reference-metadata "$REF/quant-nr-fpga/quant-reference.json"
"$BUDDY_PYTHON" -B "$MODEL/tools/prepare_model_run.py" \
  --image "$OUT/image/qwen_model.bin" --elf "$OUT/image/qwen_model.elf" \
  --weights "$BASE/weights/weights-w8a8.bin" \
  --weight-manifest "$BASE/weights/w8a8-segment.json" \
  --tokenizer "$MODEL/build/tokenizer.bin" --output "$OUT/prepared"
"$MODEL/tools/run_model.sh" "$OUT/prepared" \
  --fpga=5 --capture-seconds=2400 --startup-timeout=900
# SSH 中断时，用实际 RUN_ID 恢复原 worker：
examples/FPGA-BOSCAME/fpga_run.sh "$OUT/prepared/image.bin" \
  --fpga=5 --resume-run="$RUN_ID"
```

完整运行之后，分别验证数值、调用计数及编译链身份，再生成汇总；不能用部分
decode 的均值冒充 8 步测量。

```bash
ARCHIVE="$MODEL/validation/board/quant-opt/profile-28l-cap128-reproduce"
"$BUDDY_PYTHON" -B "$MODEL/tools/archive_model_run.py" \
  --run "$PWD/examples/FPGA-BOSCAME/build/fpga-runs/$RUN_ID" \
  --build "$BASE" --image-dir "$OUT/image" --prepared-dir "$OUT/prepared" \
  --host-graph "$BASE/combined-host-run/arrays.npz" \
  --quant-reference "$REF/quant-nr-fpga/arrays.npz" \
  --reference-metadata "$REF/quant-nr-fpga/quant-reference.json" \
  --layers 28 --prefill 16 --steps 8 --assets "$MODEL/assets/official" \
  --output "$ARCHIVE"
python3 -B "$MODEL/tools/summarize_kernel_timing.py" \
  --archive "$ARCHIVE" --clock-hz=14745600 --output "$ARCHIVE/timing"
```

输出：`timing/README.md` 为算子类别和所有特化汇总，`kernels.csv` 含调用次数及
单次平均耗时，`families.csv` 按语义分类，`per-stage-kernel.csv` 保留各步全部
测量值，`timing.json` 包含 cycles、秒数换算及证据 hash。
