# 激活动态量化优化

权重在构建时量化，激活仍须在推理时按 token 求 absmax 和 scale。参考仓库
`origin/w8a8`、`origin/kv_int8` 也采用这个区分；提前量化权重不能删除激活动态量化。
本实现借鉴它们共享 Q/K/V、gate/up 输入量化的思路，算子仍由 Triton →
triton-riscv → linalg → Buddy → RVV 生成，没有引入 C 计算替代。

## 两项独立开关

* `--share-activation-quantization`：传给 `triton_call_replace.py`、
  `lower_model_nr.py` 和 `run_graph_host.py`，同时启用 `--w8a8`。
  仅接受语义匹配的同一 RMSNorm 结果的等形状 FP32 View，且所有使用者必须是
  只读 linear 输入。不共享不同 norm、transpose、slice 或未知写入者。
  完整 28 层每图 Quantize 从 197 次降至 113 次，减少 84 次（42.64%）；
  外部 kernel 调用总数从 957 次降至 873 次。accumulator 仍各自独立并清零。
* `QWEN_TRITON_QUANT=rvv`：Triton builder 在 bufferization 前融合逐元素计算，
  随后运行 `--vectorize-quantize`，生成 16-lane abs/max/div/round/clamp/narrow。
  单位 stride、alias 检查及标量尾循环防止跨 lane 内存依赖和越界。
  `baseline` 保留原编译路径，未设置环境变量时也保持原路径。

两项优化均不改变数值契约：FP32 absmax/127，全零行 scale=1，逐元素 FP32
除法，按符号加 ±0.5，截断，饱和到 [-127,127]。没有用 `x*(1/scale)` 替代
`x/scale`，没有离线固定激活 scale，也没有添加 fast-math。

## 已完成的独立验证

* `common/nr/probes/quant_ops.S`：FPGA5 `run-be16db38f65745f4`，
  `vfabs.v/vmfge.vf/vmerge.vxm/vfmax.vf/vfmin.vf/vnsrl.wi` 组合，
  VL=1/7/16，e32/m1→e16/mf2→e8/mf4，舍入边界、正负零和 guard 全部通过，
  illegal instruction 计数为 0。通过后才加入公共 ISA 白名单。
* 六种模型形状 `16×{1024,2048,3072}`、`1×{1024,2048,3072}` 的目标
  LLVM 主机执行及 FPGA5 套件全部通过：`run-8019f65198fd46df`。
* `tests/Conversion/vectorize-quantize-execute.mlir`：动态长度 1..35、非零
  offset、全零行、不能精确二进制表示的 scale，与独立 SCF 标量参考一致。
  `vectorize-quantize-guards.mlir` 验证未知 alias/stride 及未证明的归约不会向量化。
* 仅共享量化、共享量化＋RVV 两种完整 28 层主机图：16+8，分别全部 27 项
  logits/KV 对照零误差。

同一 FPGA 镜像交替运行原版和优化版各四次；定时包含原 descriptor adapter 和
kernel，检查所有 INT8、scale 位模式、非零 offset、输入及缓冲区 guard：

| Shape | 原版平均 cycles | RVV 平均 cycles | 加速比 |
|---|---:|---:|---:|
| 16×1024 | 6,622,831 | 517,169.25 | 12.806× |
| 1×1024 | 413,280 | 32,508.5 | 12.713× |

证据：`validation/board/quant-opt/ab-16x1024/`、`ab-1x1024/`。
这些是单算子数据，不能直接当作完整模型的加速比。

## 完整 28 层验收结果

FPGA5 `run-c9c51a235a154732`，max sequence=128，真实 checkpoint，板上文本
`What is France?` 经 tokenizer 生成 16 token，prefill 后保留 KV 连续 decode 8 步。
严格归档返回 `MODEL_RUN_NUMERIC_PASS`、`FULL_LOGITS_KV_PASS`、`FIXED_TEXT_PASS`；
全部 27 项完整 logits/KV 检查通过。KV 全部零误差；首次 decode logits 最大误差
9.536743e-7，其余零误差，与原验收基线相同。9 个预测 token 全部保持一致。

| 测量范围 | 原版 | 共享＋RVV | 时间减少 |
|---|---:|---:|---:|
| Prefill 图计算 | 382.35 s | 247.92 s | 35.16% |
| Decode 图计算均值 | 76.59 s/token | 68.00 s/token | 11.21% |
| Decode 含准备、选择、KV 保留 | 80.01 s/token | 71.43 s/token | 10.72% |

按同一配置时钟 14.7456 MHz 换算，主机上传和 DDR 装载不计入。完整 launch 含
初始化、数值检查和 UART 为 847.57 s（14.13 min）。两个模型镜像均未启用逐 kernel
profiler；此表不能用于推断其余 kernel 的具体占比。

证据：[模型验收](../validation/board/quant-opt/model-28l-cap128/verification.json)、
[逐阶段时间及对照](../validation/board/quant-opt/performance.json)。
完整主机编译图的 27 项零误差记录随模型归档保存在 `references/host-run.json`。

## 复现单算子 A/B

在仓库根目录执行，先按 `model/README.md` 配好 Triton 工具链。`BASE_KERNELS`
指向已验收的原版量化 kernel 目录索引；输出必须选择新目录，不能通过 overlay
软链接覆盖既有构建。

```bash
MODEL="$PWD/examples/FPGA-BOSCAME/qwen3-0.6b/model"
BASE_KERNELS="$MODEL/build/ame-v05/cap128-kernel-overlay"
OUT="$MODEL/build/quant-opt-reproduce"
export TRITON_PYTHON=/home/chh/miniconda3/envs/boscame/bin/python
source examples/FPGA-BOSCAME/common/triton/triton-env.sh
export QWEN_TRITON_BUILD_ROOT="$OUT/rvv" QWEN_TRITON_QUANT=rvv
export QWEN_MAKE_FLAGS='RISCV_LD=/usr/bin/ld.lld-20 AME_GPR_MODE=fixed NR_COALESCE_FENCES=0'
args=()
for shape in 16x1024 16x2048 16x3072 1x1024 1x2048 1x3072; do
  args+=(--case "quantize_$shape")
done
"$TRITON_PYTHON" -B "$MODEL/../triton/build.py" "${args[@]}" --suite --jobs 2
examples/FPGA-BOSCAME/fpga_run.sh "$OUT/rvv/suite-all/suite.bin" \
  --fpga=5 --capture-seconds=120 --startup-timeout=240 \
  --completion-marker='[nr] RA returned: PASS'
for shape in 16x1024 1x1024; do
  for target in host nr; do
    flags=(); [[ "$target" != host ]] || flags+=(--host)
    python3 -B "$MODEL/optimization/build_quant_ab.py" \
      --baseline "$BASE_KERNELS/quantize_$shape" \
      --optimized "$OUT/rvv/quantize_$shape" \
      --output "$OUT/ab-$shape-$target" --repeats 4 "${flags[@]}"
  done
  examples/FPGA-BOSCAME/fpga_run.sh "$OUT/ab-$shape-nr/quant-ab.bin" \
    --fpga=5 --capture-seconds=100 --startup-timeout=240 \
    --completion-marker='[nr] RA returned: PASS'
done
# 每次用对应的实际 RUN_ID、shape 归档；checker 校验镜像及源码 hash、完整 UART。
python3 -B "$MODEL/optimization/archive_quant_ab.py" \
  --run "examples/FPGA-BOSCAME/build/fpga-runs/$RUN_ID" \
  --build "$OUT/ab-$shape-nr" --output "$MODEL/validation/board/quant-opt-reproduce/ab-$shape"
```

完整模型复现时，README 的三条图变换/lowering/host 命令已包含共享开关，
kernel 编译之前也已设置 `QWEN_TRITON_QUANT=rvv`。必须重新生成 replacement、
prefill/decode 图、adapter、静态库及镜像；不要把新 workspace ABI 与旧图混用。
模型默认范围保持完整 28 层、max sequence=128、16-token prefill＋8-step decode。
