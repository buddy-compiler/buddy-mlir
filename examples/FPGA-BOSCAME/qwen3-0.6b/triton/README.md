# Qwen3-0.6B Triton 算子

这里用真实 `@triton.jit` 编写与上一层相同的 72 个算子样例，已全部通过 FPGA5 实际执行与数值检查。模型配置、静态形状和精度范围见 [模型说明](../MODEL.md)；F32 模型语义、W8A8 部署和额外的 tile 尾部回归保持区别。

`kernels.py` 定义 14 类 Triton 内核，`cases.py` 按上一层 `metadata.json` 特化为 72 个样例。它们没有使用 Python 拼接 linalg 运算体，也没有用 C 计算替代 Triton 内核。

```text
kernels.py 中的 @triton.jit
  → Triton ASTSource / CPUBackend.make_ttir
  → kernel.ttir
  → thirdparty/triton-riscv 的 triton-shared-opt
  → kernel.linalg.mlir（linalg / tensor / memref / vector）
  → Buddy bufferization、NR AME / RVV 降低
  → LLVM IR → RISC-V 目标文件
  → 原 launch.c + 薄 ABI adapter + 公共 NR runtime
  → 单算子或 suite.bin
```

每个样例直接编译上一层原有的 `launch.c`，沿用其输入生成与逐元素 C oracle；不复制算子 C 实现。生成的 `adapter.c` 只转换 memref 描述符、枚举 Triton grid 并调用编译后的函数。构建时检查实际 LLVM 函数签名，要求每个 pointer 参数变成 `(i64 rank=0, rank-0 descriptor*)`，最后追加六个 `i32` grid/program-id 参数。

## 准备编译器

使用 [公共 Triton 环境与安装说明](../../common/triton/README.md) 安装锁定版本的 `thirdparty/triton-riscv` 和 Triton frontend。进入已安装 Triton 的 Python 环境，再从本目录执行：

```bash
source ../../common/triton/triton-env.sh
"$TRITON_PYTHON" export.py --list
```

本目录读取上一层 `make print-config` 的实际 Buddy/LLVM 工具和 C 编译选项，不另选一套工具链。需要覆盖工具时可使用已有的 `QWEN_MAKE_FLAGS`，例如将 `BUDDY_BIN` 指向支持 `target=nr-fpga` 的 Buddy 构建目录。

## 构建和运行

```bash
# 只导出一个真实 Triton 内核，保存 TTIR 和 triton-riscv MLIR。
make export CASE=matmul_1x1024x1024

# 调用从 Triton 降低出的主机内核，执行原 launcher 的数值检查。
make check CASE=matmul_1x1024x1024

# 全部 72 项逐项检查，再检查同一批内核链接出的 suite。
make check GROUP=all JOBS=4

# 生成并上板运行一个算子。
make run CASE=matmul_1x1024x1024 FPGA=5

# 生成全部 72 个算子和一个公共 suite。
make all JOBS=4
make run GROUP=all FPGA=5
```

分组为 `ame`（12 个整数 matmul，含尾部回归）、`rvv`（11 个 F32 线性形状）、`aux`（其余 49 个）和 `smoke`。`scalar` 是 `rvv` 的兼容别名，实际 NR 代码使用 Buddy RVV matmul 降低。可以直接传多个 `--case` 给 `build.py`，再加 `--suite` 链接成一个镜像。

所有编译产物位于 `build/`，例如：

```text
build/matmul_1x1024x1024/
  kernel.ttir
  kernel.linalg.mlir
  adapter.c
  frontend.json
  host/check
  host/output.log
  nr/kernel.ll
  nr/kernel.nr.S
  nr/matmul_1x1024x1024.bin
  nr/matmul_1x1024x1024.audit.json
  nr/manifest.json
build/suite-all/suite.bin
build/suite-all/suite.audit.json
build/suite-all/manifest.json
```

`frontend.json` 记录 Triton 函数、类型、constexpr、grid 和 TTIR/MLIR 哈希；`manifest.json` 记录 LLVM、原 launcher、adapter、镜像和实际工具配置。suite 的 manifest 同时列出全部 case、公共编译/指令工具及运行时源码哈希，供上板结果对照。每个 NR 镜像在生成 `.bin` 前必须通过公共 `check_nr_elf.py`：遍历整个链接 ELF 的可执行字节，检查 AME 编码、RVV 白名单、向量 CSR 和访存前后 fence，并保存完整审计报告。

## NR 降低和验证

整数 matmul 使用有符号 `tl.dot`、I32 累加。常规模型形状整除 tile 时省略无效的 mask，使输入保持 memref 视图，避免反复 padding 和矩阵搬运；NR pass 负责切分硬件 K=64 tile。`matmul_3x19x70` 保留完整边界 mask。F32 dot 和 attention 使用实际 Buddy 向量化后的 LLVM 额外执行一次主机 oracle 检查。

临时 tensor buffer 通过 `promote-buffers-to-stack` 进入当前 grid 函数的栈；构建拒绝仍含 `malloc/free` 的最终 LLVM，避免词表 matmul 的数千次 grid 调用耗尽裸机 bump allocator。strided memref copy 通过真实 Buddy pass 降为循环。NR 汇编继续经过公共编码器和严格指令检查，禁止未经允许的指令、`vlenb` 和 whole-register 操作。

仅生成的 Triton 内核启用 RVV；C launcher 和 ABI adapter 使用 `rv64gc_zicbom`，避免描述符初始化自动生成板卡不支持的向量指令。F32 线性矩阵继承公共 `RVV_LINEAR_FLAGS`，禁用会引入 whole-register copy 的 LLVM machine LICM/CSE；attention 和其他内核使用公共 `RVV_FLAGS` 的默认机器优化。两者仅设置最小 VLEN=512，不指定最大 VLEN，避免编译器生成 whole-register load/store；最终 ELF 审计会拒绝这些指令。

已保存的 [主机验证记录](validation/host/verification.json) 包含 72 个单例、72 项链接 suite，以及 15 个实际 Buddy 向量化后 LLVM 的数值检查，全部通过；相邻日志和 manifest 记录对应 LLVM 与可执行文件哈希。[代码生成审计汇总](validation/codegen.json) 记录全部 72 个 ELF 和最终 suite 的检查、源码及镜像哈希，[完整 suite 审计](validation/elf-audit.json) 列出最终 ELF 的实际指令种类。

[最终 FPGA5 验证](validation/all/verification.json) 的 72 项全部通过，包含完整词表线性矩阵、prefill/decode attention 和 AME 尾部回归。对应运行 `run-1ee9bd4b8bea48c5`，镜像 SHA256 为 `b6c5f4749239d11cd93d26f6062b75159c87c07184c03599dcc16a990d76dc7f`；[原始串口日志](validation/all/uart.raw.log)、上传/DDR 读回结果及 manifest 一并保留。复现与记录方法见 [验证说明](validation/README.md)。测试使用原 launcher 的确定性输入与独立 oracle；不需要下载 checkpoint，也不代表已运行完整 Qwen3 文本生成。

前端调用方法依据 ModelZoo 的 `examples/tools/triton_lower.py` 和当前 triton-riscv API。用户提供的 `CBalaa/BOSC-FPGA-MLIR/tree/main/examples/triton` 链接在本次调查时，GitHub 页面、API 和 raw URL 均返回 404，因此这里不声称读取或复用了该仓库代码；这不影响使用已验证的真实 Triton 前端。
