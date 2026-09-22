# Qwen3-0.6B：独立算子与整模型到 NR FPGA

本目录包含 Qwen3-0.6B 的 72 个独立算子样例。每个目录提供 `kernel.mlir`、
`launch.c`、`metadata.json` 和 `makefile`：使用 Buddy lowering 生成算子对象，
与 C launch 和公共 NR 运行时链接成可上板的裸机二进制。输入在板上生成，
实际算子结果与独立 C 参考逐元素比较。

[MODEL.md](MODEL.md) 记录完整结构、来源、算子列表和形状推导；
[model.json](model.json) 是机器可读配置。上述算子样例独立验证；
[model/](model/README.md) 另提供集成真实权重、tokenizer、KV cache 的完整模型。
参考仓库仅供结构/平台查证，构建和上传不依赖 `references/`。

## 构建 none 整模型二进制

整模型使用 `model/tools/build_nr_w8a8_image.py`，不是下面单算子的 `make all`。
这里的 `--ame-startup=none` 不执行额外启动同步，但保留图结束后的
`--graph-sync=ame-resync`。与已验收 `production-control` 基线一致的运行配置还包括
4 MiB append-only console、实时 NH drain，以及关闭 profiling、hang-watch、
intermediate probe、graph-entry fence 和 workspace CBO 诊断。

下面从**只有仓库源码、没有任何构建目录或模型资源**开始，构建完整 28 层、
batch=1、cache capacity=128、16-token prefill、8 步 decode、完整词表 151936
的模型。LLVM、Buddy Python bindings、Triton、46 个 kernel、计算图、权重、
tokenizer、数值参考和最终镜像全部在下面生成；不读取历史 `build/`、验收归档
或 `references/` 作为构建输入，也不修改 `examples/BuddyQwen3`。

各段按顺序在**同一个 Bash 会话**执行。第 1～8 步只在本机构建；第 9 步才上板。
这会按当前源码重建 none 配置，**不保证与历史 `production-control` 逐字节相同，
也不能沿用它的 PASS**。若只想运行已验收的精确镜像，见
[公共 README 的原始 none 指令](../README.md#上传并运行qwen3-原始-none-镜像)。

### 1. 系统依赖、源码和新的工作目录

宿主机使用 Linux x86-64，无需 GPU。先安装 Git、C/C++ 编译器、GNU Make、
pkg-config、curl、unzip、zlib/libffi 开发包，以及 **Python 3.11 和 3.12**
的解释器、开发头文件与 venv 支持。Debian/Ubuntu 的公共依赖可安装为：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git pkg-config curl unzip \
  ca-certificates zlib1g-dev libffi-dev
```

Python 软件包是否同时提供这两个版本取决于发行版；请先通过所在系统的软件源
安装相应的 `python3.11` / `python3.11-dev` / `python3.11-venv` 和
`python3.12` / `python3.12-dev` / `python3.12-venv`，或使用自行安装的对应解释器。
下面会创建全新 venv，**不要求已有 conda、boscame 或 qwen3fpga 环境**。
需能下载 GitHub、Python 包和 Hugging Face 资源；Triton 子模块使用 GitHub SSH URL，
需先配置可用的 GitHub SSH 认证。完整工具链及中间文件较大，建议预留至少
150 GiB 磁盘、64 GiB 内存，并按机器资源调低 `JOBS`；这些是建议配置，不是最低要求。

从仓库根目录开始：

```bash
set -euo pipefail
REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"
MODEL="$REPO/examples/FPGA-BOSCAME/qwen3-0.6b/model"
TRITON_COMMON="$REPO/examples/FPGA-BOSCAME/common/triton"
JOBS=8

# 这两个固定路径是部分模型脚本的约定；下面会从源码创建，不是预置输入。
# 若已存在，停止并另选干净源码工作副本，不删除/覆盖原来的构建。
for directory in "$REPO/llvm/build-2d26" "$REPO/build-migrate"; do
  [[ ! -e "$directory" && ! -L "$directory" ]] || {
    printf '已有工具链目录，请在干净源码工作副本执行：%s\n' "$directory" >&2
    exit 1
  }
done

# 只初始化这条流水线所需的源码子模块；不递归下载其他平台的工具链。
git submodule update --init llvm thirdparty/triton-riscv

# 父目录也由本步骤创建；mktemp 保证本次产物不覆盖历史模型。
mkdir -p "$MODEL/build"
OUT="$(mktemp -d "$MODEL/build/none-from-scratch.XXXXXX")"
printf '保存本次工作目录（后续所有步骤使用它）：%s\n' "$OUT"
python3.11 -m venv "$OUT/venv-buddy"
python3.12 -m venv "$OUT/venv-triton"
export BUDDY_PYTHON="$OUT/venv-buddy/bin/python"
export TRITON_PYTHON="$OUT/venv-triton/bin/python"
export PATH="$OUT/venv-buddy/bin:$PATH"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
"$BUDDY_PYTHON" -m pip install --upgrade pip
"$TRITON_PYTHON" -m pip install --upgrade pip
"$BUDDY_PYTHON" -m pip install cmake==3.31.10 ninja==1.13.2
# 先选 CPU 版，避免下载无关的 CUDA 运行时。
"$BUDDY_PYTHON" -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu
"$BUDDY_PYTHON" -m pip install \
  -r "$REPO/requirements.txt" -r "$REPO/llvm/mlir/python/requirements.txt"

# 在开始耗时编译前核对源码版本；不自动 reset 不匹配的源码。
"$BUDDY_PYTHON" - "$REPO" "$TRITON_COMMON/toolchain-lock.json" <<'PY'
import json, pathlib, subprocess, sys
repo = pathlib.Path(sys.argv[1])
lock = json.loads(pathlib.Path(sys.argv[2]).read_text())
for name, relative in (("llvm", "llvm"), ("triton_riscv", "thirdparty/triton-riscv")):
    actual = subprocess.check_output(
        ["git", "-C", str(repo / relative), "rev-parse", "HEAD"], text=True).strip()
    assert actual == lock[name]["commit"], (name, actual, lock[name]["commit"])
print("toolchain source revisions: PASS")
PY
```

### 2. 从源码编译 LLVM / MLIR / Buddy（含 Python bindings）

不能用普通发行版 LLVM 替代本仓库 LLVM。Buddy 与 MLIR Python bindings 都使用
上一步的 Python 3.11；仅构建 `buddy-opt` 而关闭 bindings 无法导入整个模型。
OpenMP 和 MLIR runner 动态库用于后面的主机计算图验证，不链接进 FPGA 裸机镜像。

```bash
cmake -S "$REPO/llvm/llvm" -B "$REPO/llvm/build-2d26" -G Ninja \
  -DLLVM_ENABLE_PROJECTS='mlir;clang;lld' \
  -DLLVM_ENABLE_RUNTIMES=openmp -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DLLVM_TARGETS_TO_BUILD='host;RISCV' \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=1 -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$BUDDY_PYTHON" -DPython_EXECUTABLE="$BUDDY_PYTHON"
cmake --build "$REPO/llvm/build-2d26" --parallel "$JOBS"

cmake -S "$REPO" -B "$REPO/build-migrate" -G Ninja \
  -DMLIR_DIR="$REPO/llvm/build-2d26/lib/cmake/mlir" \
  -DLLVM_DIR="$REPO/llvm/build-2d26/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON \
  -DPython3_EXECUTABLE="$BUDDY_PYTHON" -DPython_EXECUTABLE="$BUDDY_PYTHON"
cmake --build "$REPO/build-migrate" --parallel "$JOBS"
# 显式确保两套 Buddy Python 包齐全。
cmake --build "$REPO/build-migrate" --parallel "$JOBS" \
  --target python-package-buddy python-package-buddy-mlir

export PYTHONPATH="$REPO/build-migrate/python_packages"
export LLVM_SYSPATH="$REPO/llvm/build-2d26"
export LLVM_BIN="$LLVM_SYSPATH/bin"
export BUDDY_MLIR_BINARY_DIR="$REPO/build-migrate/bin"
LINKER="$LLVM_BIN/ld.lld"
"$LINKER" --version
"$BUDDY_PYTHON" -c \
  'from buddy_mlir import ir, execution_engine; from buddy.compiler.frontend import DynamoCompiler; print("Buddy Python: PASS")'
```

LLD 要求 >= 20；这里使用本次编译出的 `ld.lld`，不依赖系统的 `ld.lld-20`。

### 3. 编译锁定的 Triton 前端

Triton 使用另一套 Python 3.12 环境。安装脚本根据
[`toolchain-lock.json`](../common/triton/toolchain-lock.json) 下载嵌套 Triton 源码、
验证版本、应用锁定的兼容补丁并编译，最后执行前端 smoke test。
它复用第 2 步刚生成的 LLVM/Buddy，不下载另一套 Buddy。

```bash
export BUDDY_SOURCE_DIR="$REPO"
export TRITON_RISCV_DIR="$REPO/thirdparty/triton-riscv"
export TRITON_DIR="$TRITON_RISCV_DIR/triton"
unset TRITON_SHARED_OPT_PATH
"$TRITON_COMMON/setup-triton.sh" --jobs="$JOBS"
source "$TRITON_COMMON/triton-env.sh"
"$TRITON_COMMON/setup-triton.sh" --check
```

如果源码 revision 不符合 lock，脚本会停止并保留现场；不要用 `git reset` 强行覆盖。

### 4. 下载官方资源并生成 tokenizer

资源锁定官方 revision `c1899de289a04d12100db370d81485cdf75e47ca`。
checkpoint 约 1.50 GB，并校验 SHA256。所有资源写入本次 `$OUT`，不要求
已有 `model/assets/` 或 `model/build/tokenizer.bin`。

```bash
"$BUDDY_PYTHON" -B "$MODEL/tools/fetch_assets.py" --output "$OUT/assets"
"$BUDDY_PYTHON" -B "$MODEL/tools/fetch_checkpoint.py" \
  --output "$OUT/checkpoint" --manifest "$OUT/checkpoint-manifest.json" \
  --expected-sha256 f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
"$BUDDY_PYTHON" -B "$MODEL/tools/pack_tokenizer.py" \
  --assets "$OUT/assets" --output "$OUT/tokenizer.bin"

PROMPT=151644,872,198,3838,374,9625,30,151645,198,151644,77091,198,151667,271,151668,271
COMMON=(--assets "$OUT/assets" --checkpoint "$OUT/checkpoint"
        --layers 28 --prefill-len 16 --max-cache-len 128)
```

### 5. 生成模型专用 case 并编译全部 46 个 kernel

下面直接列出完整清单，**不从历史 archive.json 提取清单或对象文件**。
共享 case 的源码已在本目录；cache=128 的模型专用 case 在本次生成。
11 个 INT8 linear 使用 AME N64 / direct GPR / fence 合并，7 个反量化算子和
base 组中的 6 个动态量化算子使用 RVV。

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

AME_CASES=(
  matmul_16x1024x1024 matmul_16x1024x2048 matmul_16x1024x3072
  matmul_16x2048x1024 matmul_16x3072x1024
  matmul_1x1024x1024 matmul_1x1024x2048 matmul_1x1024x3072
  matmul_1x151936x1024 matmul_1x2048x1024 matmul_1x3072x1024
)
RVV_CASES=(
  dequantize_16x1024 dequantize_16x2048 dequantize_16x3072
  dequantize_1x1024 dequantize_1x151936 dequantize_1x2048 dequantize_1x3072
)
BASE_CASES=(
  attention_pv_position_16x16x128x128 attention_pv_position_16x1x128x128
  attention_qk_position_native_16x16x128x128 attention_qk_position_native_16x1x128x128
  attention_scale_mask_position_16x16x128 attention_scale_mask_position_16x1x128
  embedding_w8a8_16x1024 embedding_w8a8_1x1024
  kv_cache_update_position_16x8x128_cap128 kv_cache_update_position_1x8x128_cap128
  layout_context_8x16x128 layout_context_8x1x128
  quantize_16x1024 quantize_16x2048 quantize_16x3072
  quantize_1x1024 quantize_1x2048 quantize_1x3072
  rmsnorm_128x128 rmsnorm_16x1024 rmsnorm_16x128
  rmsnorm_1x1024 rmsnorm_256x128 rmsnorm_8x128
  silu_16x3072 silu_1x3072 softmax_16x16x128 softmax_16x1x128
)
[[ ${#AME_CASES[@]} == 11 && ${#RVV_CASES[@]} == 7 && ${#BASE_CASES[@]} == 28 ]]

export QWEN_TRITON_QUANT=rvv
for group in base ame rvv; do
  case "$group" in
    base) names=("${BASE_CASES[@]}") ;;
    ame)  names=("${AME_CASES[@]}") ;;
    rvv)  names=("${RVV_CASES[@]}") ;;
  esac
  args=()
  for name in "${names[@]}"; do args+=(--case "$name"); done
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

每个 case 都经过真实 Triton → triton-riscv → linalg → Buddy lowering。
`--host` 生成主机验证版本，不运行 FPGA 指令；overlay 只汇集本次分组构建的产物。
不要将后续编译输出指向 overlay 中的软链接。

### 6. 导入完整模型、替换 kernel 调用并 lowering

```bash
"$BUDDY_PYTHON" -B "$MODEL/tools/import_model.py" "${COMMON[@]}" \
  --output "$OUT/import" --fuse none --save-mlir --init-cache
"$BUDDY_PYTHON" -B "$MODEL/tools/weight_layout.py" \
  --mlir "$OUT/import/forward_decode.mlir" --params "$OUT/import/params.json" \
  --checkpoint "$OUT/checkpoint/model.safetensors" \
  --config "$OUT/assets/config.json" --output "$OUT/import/weight-layout.json"
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

这两个库构建脚本会重新创建其输出目录，因此 `OUT=` 必须保持为上述本次工作目录下的
`model-lib` / `host-bridge` **叶目录**，不能传仓库根目录或整个 `$OUT`。
应生成 `nr-prefill/forward_prefill.ll`、`nr-decode/forward_decode.ll`、
`model-lib/libqwen_triton.a` 和 `host-bridge/libqwen_triton_host.so`。
每图预期有 873 个外部调用、未覆盖的大计算为 0；共享激活量化报告为
`created=113, reused=84`。静态库包含 46 个 kernel 及其 adapter，
图、report、adapter、KV 容量和权重 layout 不得与其他构建混用。

### 7. 生成独立参考并验证主机计算图

参考数据也从本次 checkpoint 计算，不使用旧的 `arrays.npz`。
FP32 参考用于核对量化轨迹；`nr-fpga` 与 `triton-host` 两种算术 profile 分别用于
板上和主机验证。下面的 `--capture-intermediates` 仅用于主机验证，不启用板上 probe。

```bash
"$BUDDY_PYTHON" -B "$MODEL/tools/host_reference.py" \
  --assets "$OUT/assets" --checkpoint "$OUT/checkpoint" \
  --layers 28 --max-cache-len 128 --prompt-ids "$PROMPT" --decode-steps 8 \
  --output "$OUT/fp32-reference"
for profile in nr-fpga triton-host; do
  "$BUDDY_PYTHON" -B "$MODEL/tools/quant_model_reference.py" \
    --assets "$OUT/assets" --checkpoint "$OUT/checkpoint" \
    --layout "$OUT/import/weight-layout.json" --layers 28 --max-cache-len 128 \
    --prompt-ids "$PROMPT" --decode-steps 8 --arithmetic-profile "$profile" \
    --fp32-reference "$OUT/fp32-reference" --output "$OUT/quant-$profile"
done
"$BUDDY_PYTHON" -B "$MODEL/tools/run_graph_host.py" \
  --assets "$OUT/assets" --checkpoint "$OUT/checkpoint" \
  --layers 28 --max-cache-len 128 --output "$OUT/host-run" \
  --layout "$OUT/import/weight-layout.json" --triton-build "$OVERLAY" \
  --external-lib "$OUT/host-bridge/libqwen_triton_host.so" \
  --replace --attention --attention-position --attention-native-key --w8a8 --share-activation-quantization \
  --decode-steps 8 --capture-intermediates --prompt-ids "$PROMPT" \
  --quant-reference-dir "$OUT/quant-triton-host"
```

必须先确认主机验证通过，再继续打包。主机通过不能替代 FPGA 验收。

### 8. 打包 W8A8 权重，构建 none ELF/BIN，准备 DDR 文件

```bash
"$BUDDY_PYTHON" -B "$MODEL/tools/build_nr_w8a8_segment.py" \
  --report "$OUT/replacement/triton-call-replacement.json" \
  --layout "$OUT/import/weight-layout.json" \
  --checkpoint "$OUT/checkpoint/model.safetensors" \
  --config "$OUT/assets/config.json" --output "$OUT/weights"
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
  --tokenizer-blob "$OUT/tokenizer.bin" \
  --reference-arrays "$OUT/quant-nr-fpga/arrays.npz" \
  --reference-metadata "$OUT/quant-nr-fpga/quant-reference.json" \
  --ame-startup=none --graph-sync=ame-resync --ame-cache-sync=none \
  --console-mode=append-only --console-capacity=4194304 --console-drain=live

"$BUDDY_PYTHON" -B "$MODEL/tools/prepare_model_run.py" \
  --image "$OUT/image/qwen_model.bin" --elf "$OUT/image/qwen_model.elf" \
  --weights "$OUT/weights/weights-w8a8.bin" \
  --weight-manifest "$OUT/weights/w8a8-segment.json" \
  --tokenizer "$OUT/tokenizer.bin" --output "$OUT/prepared"

sha256sum "$OUT/image/qwen_model.elf" "$OUT/image/qwen_model.bin" \
  "$OUT/prepared/image.bin" "$OUT/weights/weights-w8a8.bin" "$OUT/tokenizer.bin" \
  | tee "$OUT/SHA256SUMS"
printf '本次 none 构建目录：%s\n' "$OUT"
```

**至此完整模型二进制构建完成，尚未操作 FPGA。** 最终文件均位于本次 `$OUT`：

- `image/qwen_model.elf`、`image/qwen_model.bin`、`image/qwen_model.map`：裸机程序与符号布局。
- `weights/weights-w8a8.bin`、`tokenizer.bin`：独立装载的真实模型权重和 tokenizer。
- `prepared/`：补齐的 boot image、权重、tokenizer 及 `ddr-load.plan`，上板使用这个目录。

检查 `image/image.json` 的 `status` 为 `PASS`、`undefined_symbols` 为空，
`image/elf-audit.json` 为 `PASS`；`w8a8-image-plan.json` 应记录
`ame_startup: none`、`graph_completion_sync: ame-resync` 和上述 console 配置。
不要只上传裸机 BIN 而漏掉 DDR 权重和 tokenizer。参考数组仅用于运行后比较，
不代替模型推理、token 选择或 KV 更新。

### 9. 可选：FPGA5 运行和严格验收

只有需要上板时才执行本节。先确认 FPGA5 空闲、SSH 别名 `fpga` 支持免交互认证，
服务器指定目录已有 UVHS / 平台 Makefile，且没有其他 UART owner。所有远端操作限定在
`/home/hjuser/Desktop/fpga-tester-ISCAS`，不修改其软链接指向的外部目录。

```bash
"$MODEL/tools/run_model.sh" "$OUT/prepared" \
  --fpga=5 --remote-dir=/home/hjuser/Desktop/fpga-tester-ISCAS \
  --capture-seconds=1800 --startup-timeout=900
```

保存脚本打印的实际 `run-ID`。`Ctrl+C` 或 SSH 断开不保证远端 worker / FPGA 已停止；
中断与恢复说明见[公共运行文档](../README.md#上传并运行qwen3-原始-none-镜像)。
若更换 Bash 会话，需恢复 `REPO`、`MODEL`、**本次实际的** `OUT` 和
`BUDDY_PYTHON="$OUT/venv-buddy/bin/python"`；不要重新运行第 1 步生成另一个目录。

运行完成后，用本次 build 和 run 进行归档（将占位 ID 替换为实际值）：

```bash
RUN_ID=run-ACTUAL_ID
"$BUDDY_PYTHON" -B "$MODEL/tools/archive_model_run.py" \
  --run "$REPO/examples/FPGA-BOSCAME/build/fpga-runs/$RUN_ID" \
  --build "$OUT" --image-dir "$OUT/image" --prepared-dir "$OUT/prepared" \
  --host-graph "$OUT/host-run/arrays.npz" \
  --quant-reference "$OUT/quant-nr-fpga/arrays.npz" \
  --reference-metadata "$OUT/quant-nr-fpga/quant-reference.json" \
  --layers 28 --prefill 16 --steps 8 --assets "$OUT/assets" \
  --output "$OUT/board-validation/$RUN_ID"
```

要求 27 项 logits/KV、固定文本校验、DDR 回读以及 `RA returned: PASS` 全部通过，
严格归档输出 `MODEL_RUN_NUMERIC_PASS`。只看到文本、编译成功或旧的 PASS 记录均不够。
更详细的数值契约与产物说明见 [model/README.md](model/README.md)。

## 覆盖与执行位置

采用 batch=1、prefill S=16、decode S=1/T=17；隐藏维度1024、中间维度3072，
16个Q头、8个KV头、每头128。28层复用相同形状。完整词表为151936；
lm_head只计算最后一个token，因此M=1。

| 目录类型 | 数量 | lowering / 实际执行 |
| --- | ---: | --- |
| `matmul_MxNxK` | 11 | 有符号 i8 × i8 → i32，NR AME |
| `matmul_3x19x70` | 1 | AME tile 尾部回归，非模型形状 |
| `matmul_MxNxK_f32` | 11 | 同模型线性公式，Buddy transpose-B RVV FP32 |
| `attention_qk_*`、`attention_pv_*` | 4 | `linalg.batch_matmul`，Buddy RVV FP32 |
| embedding、RMSNorm、RoPE、softmax、SiLU、mul/add、mask、KV、GQA、layout | 32 | `linalg.generic` / `linalg.fill`，RA |
| quantize、dequantize | 13 | 显式 W8A8 部署辅助算子，RA |

官方模型是 BF16；FP32 样例验证公式和维度，W8A8 是额外的部署变换。
不能据此宣称 BF16 整模型或量化后文本质量已经验证。

## 构建和单个算子运行

在仓库根目录执行，例如：

```bash
make -C examples/FPGA-BOSCAME/qwen3-0.6b/matmul_1x1024x1024 all check
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/qwen3-0.6b/matmul_1x1024x1024/build/matmul_1x1024x1024.bin \
  --fpga=5 --capture-seconds=900 --completion-marker='[nr] RA returned:'
```

也可在算子目录执行 `make run FPGA=5`。`check` 将同一份 linalg 降为主机循环，
链接相同 launch 中的独立参考做数值检查；15个FP32矩阵/attention样例还检查
实际Buddy向量化后的LLVM，以覆盖向量降低本身。这些检查不替代 FPGA 测试。
必须看到对应 `verify <算子>: PASS` 和 `[nr] RA returned: PASS`。
脚本验证上传哈希与DDR读回，缺少完成标记、数值失败、trap或超时返回非零。

`build/` 保留 `.ame.mlir`、`.llvm.mlir`、`.ll`、`.s`、`.nr.S`、算子 `.o`、
launch `.o`、最终 `.elf` / `.map` / `.bin`，便于逐阶段检查。

## 批量构建与验证

```bash
cd examples/FPGA-BOSCAME/qwen3-0.6b
make -j4 all
make check
make suite GROUP=all
make run GROUP=all FPGA=5 CAPTURE_SECONDS=3600
```

组名：`aux`（49个）、`ame`（12个）、`rvv`（11个FP32线性算子）、`smoke`（4个）、`all`（72个）。
`scalar` 保留为 `rvv` 的兼容组名，实际使用RVV。
集合镜像复用同一640MiB NOLOAD workspace，每个case重新初始化输入；
仍调用各样例相同的算子目标文件和launch，不替换实现。FP32完整词表矩阵较慢，
可先运行 `make run GROUP=smoke`。完整运行时间受板上时钟影响。

```bash
python3 tools/build_suite.py --case matmul_3x19x70 --case matmul_1x1024x1024
make suite-host GROUP=aux
```

`tools/record_run.py` 对集合manifest与脚本输出进行核对，检查每个case的成功标记，
保存运行证据；使用示例见 [validation/README.md](validation/README.md)。

## NR lowering 与指令约定

流水线的完整命令在 [common.mk](common.mk)。核心过程为：

```text
linalg buffer IR
  → --lower-linalg-to-boscame=target=nr-fpga
  → --lower-bosc-ame
  → linalg/scf/math/memref/func 到 LLVM dialect
  → buddy-translate --buddy-to-llvmir
  → FP32矩阵/attention使用Buddy向量化pass；其余转换为循环
  → llc (RISC-V, lp64d, medany，RVV使用512-bit固定向量、VL≤16)
  → AME 指令编码 + NR ISA 检查/fence
  → clang 生成 kernel.o
  → kernel.o + launch.o + NR runtime → ELF → 完整ELF指令审计 → raw BIN
```

`nr-fpga` 直接加速静态正维度、buffer形式的 signed i8×i8→i32 `linalg.matmul`，
按16×16×64切块并处理尾部。B可用物理 `[N,K]`、逻辑 `[K,N]` 的strided memref，
由普通 `mlbe8.m` 读取；普通 `[K,N]` 输入由pass打包小tile。
C保留输入累加值，`msce32.m` 写回原始i32；反量化必须显式 `arith.sitofp`。
FP32等其他类型保留给后续linalg lowering；这里对FP32矩阵与attention显式执行
Buddy的transpose-B或batchmatmul向量化pass。GEM5的默认 `upstream` profile保持独立。
旧 `qwen3-fpga` 的f32-store契约不用于本目录。

NR不支持旧同步使用的 `mlbte8.m`。公共NR编码工具检查指令，并在AME指令前后
插入fence；每个整数launch在调用完整kernel前做一次ModelZoo的AME状态同步。
NH从 `0x80000000` 启动RA；仅NH访问UART。大workspace放在 `0xb8000000`，
避开已知故障地址区间，NOLOAD确保上传镜像不会包含几百MiB零字节。
平台实现与来源见 [../common/nr/README.md](../common/nr/README.md)。

FP32线性层的RVV代码生成使用 `-disable-machine-licm -disable-machine-cse`，避免LLVM将向量
清零外提后插入当前板卡行为不正确的 `vmv1r.v`。整寄存器复制/访存和不可用的
vector CSR读取均由指令审计拒绝。C launch、ABI adapter和运行时以
`rv64gc_zicbom` 编译；单独关闭C循环向量化不足以禁止描述符初始化使用RVV。
attention保留已经板上验证的默认机器码优化；两类选项分别由 `RVV_FLAGS` 和
`RVV_LINEAR_FLAGS` 配置。硬件探测与修复依据见 [validation](validation/README.md)。

## 工具链

使用当前仓库编译的 Buddy 与 LLVM；普通发行版 LLVM 不包含 BOSCAME 后端。
从零安装、源码编译及 Python bindings 配置见上面的
[整模型构建步骤](#构建-none-整模型二进制)第 1～3 步；同一套工具链也可用于独立算子。
不要为整模型关闭 `BUDDY_MLIR_ENABLE_PYTHON_PACKAGES`。

可覆盖 `BUDDY_BIN`、`BUDDY_OPT`、`BUDDY_TRANSLATE`、`LLVM_BIN`、`RISCV_CC`、
`RISCV_LD`、`RISCV_OBJCOPY`、`HOST_CC`。没有硬编码机器绝对路径。
生成器用于维护静态样例：`make generate`；正常构建直接读取已保存的MLIR/C，
不需要PyTorch、transformers或模型权重。

## Triton 前端

[triton/](triton/README.md) 提供同样72个样例的真实 `@triton.jit` 实现，
经 `triton-riscv` 生成linalg后接入相同Buddy后端、C校验与NR运行时。
公共安装脚本和锁定版本见 [common/triton](../common/triton/README.md)。
