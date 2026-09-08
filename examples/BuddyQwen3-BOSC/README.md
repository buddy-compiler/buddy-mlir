# BuddyQwen3-BOSC

Buddy 前端（PyTorch / Transformers Qwen3-0.6B）→ TOSA/Linalg MLIR → BOSC/AME
W8A8 → RISC-V 裸机 FPGA 镜像。本 README 只说明如何构建出可上板执行的 `.bin`。

## 前置条件

1. buddy-mlir 已构建出以下工具与 Python 包（相对 buddy-mlir 根目录）：
   - `build/bin/buddy-opt`、`build/bin/buddy-translate`
   - `build/python_packages`（`buddy.compiler` + `buddy_mlir`）
   - `llvm/build/bin/llc`、`llvm/build/bin/mlir-opt`

2. 环境变量 `QWEN3_0_6B_DIR` 指向 qwen3-0.6b 检出目录（需包含
   `models/qwen3/model_data`、`platform`、`tokenizer/data`）。

3. 导入阶段需要带 `torch` 与 `transformers` 的 Python 环境；通过环境变量
   `PYTHON_BIN` 指定解释器，不设置时使用 `python3`。

4. RISC-V 工具链 `riscv64-linux-gnu-gcc` / `-ld` / `-objcopy`；找不到时
   Makefile 自动回退到 clang / ld.lld。

## 构建

在 `examples/BuddyQwen3-BOSC` 目录下依次执行：

```bash
export QWEN3_0_6B_DIR=<qwen3-0.6b 检出目录>

# 1. 导入：torch → MLIR + W8A8 参数（产出 5 个组件图与参数文件）
./run_layerwise_import.sh \
  --output-dir build-model-w8a8-layerwise-padded128 \
  --prefill-len 128 --max-cache-len 256 \
  --prompt 'What is 1+1?'

# 2. Lower：MLIR → LLVM IR
./run_layerwise_lower.sh build-model-w8a8-layerwise-padded128

# 3. 汇编链接：LLVM IR → .bin（自动运行镜像校验，PASS 即成功）
make -j8 layerwise-image-check \
  LAYERWISE_BUILD=build-model-w8a8-layerwise-padded128 \
  LAYERWISE_IMAGE_BUILD=build-layerwise-fpga \
  LAYERWISE_DECODE_STEPS=2
```

清理构建产物：

```bash
make clean
```

## 产物

- `build-layerwise-fpga/buddy_qwen3_layerwise.bin` —— 上板执行的扁平二进制。
- `make` 结束时会打印烧写命令，形如：

  ```text
  writemem -rtl ariane_xilinx.i_ddr[N:0] -file .../buddy_qwen3_layerwise.bin -file_type bin
  ```

  按打印出的地址与文件名烧写即可。

## 常用参数

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--prefill-len` / `--max-cache-len` | 固定 prompt 长度 / KV cache 容量 | 128 / 256 |
| `--prompt` | 固定提示词（左填充到 `--prefill-len`） | - |
| `LAYERWISE_DECODE_STEPS` | 镜像内的 decode 步数 | 1 |
| `LAYERWISE_HEAP_SIZE` | 裸机运行时堆大小 | 50331648u |
