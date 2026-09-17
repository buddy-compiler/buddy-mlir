# Qwen3-0.6B：linalg 算子到 NR FPGA

本目录包含 Qwen3-0.6B 的 72 个独立算子样例。每个目录提供 `kernel.mlir`、
`launch.c`、`metadata.json` 和 `makefile`：使用 Buddy lowering 生成算子对象，
与 C launch 和公共 NR 运行时链接成可上板的裸机二进制。输入在板上生成，
实际算子结果与独立 C 参考逐元素比较。

[MODEL.md](MODEL.md) 记录完整结构、来源、算子列表和形状推导；
[model.json](model.json) 是机器可读配置。这里验证独立算子，未集成权重或整模型生成。
参考仓库仅供结构/平台查证，构建和上传不依赖 `references/`。

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

使用当前仓库编译的Buddy与LLVM；普通发行版LLVM不包含BOSCAME后端。
可用已有的 `boscame` conda环境。第一次构建需使LLVM库、TableGen工具和源码版本一致：

```bash
conda activate boscame
cmake -S llvm/llvm -B llvm/build-2d26 -G Ninja \
  -DLLVM_ENABLE_PROJECTS='clang;mlir;lld' -DLLVM_TARGETS_TO_BUILD='RISCV;X86' \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build llvm/build-2d26 --target clang llc opt mlir-opt mlir-translate llvm-objcopy llvm-objdump ld.lld
cmake -S . -B build-migrate -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=OFF \
  -DMLIR_DIR="$PWD/llvm/build-2d26/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/llvm/build-2d26/lib/cmake/llvm"
cmake --build build-migrate --target buddy-opt buddy-translate
```

可覆盖 `BUDDY_BIN`、`BUDDY_OPT`、`BUDDY_TRANSLATE`、`LLVM_BIN`、`RISCV_CC`、
`RISCV_LD`、`RISCV_OBJCOPY`、`HOST_CC`。没有硬编码机器绝对路径。
生成器用于维护静态样例：`make generate`；正常构建直接读取已保存的MLIR/C，
不需要PyTorch、transformers或模型权重。

## Triton 前端

[triton/](triton/README.md) 提供同样72个样例的真实 `@triton.jit` 实现，
经 `triton-riscv` 生成linalg后接入相同Buddy后端、C校验与NR运行时。
公共安装脚本和锁定版本见 [common/triton](../common/triton/README.md)。
