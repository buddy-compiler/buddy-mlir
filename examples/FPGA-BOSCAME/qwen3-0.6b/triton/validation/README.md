# Triton 算子验证记录

这里保存真实 Triton 前端、Buddy 降低和 FPGA 执行的验证资料。输入与数值 oracle 来自上一层原有 `launch.c`。覆盖 72 个静态样例，不包含 checkpoint 推理或完整模型文本生成。

- `host/`：72 个单例的主机输出、72 项链接 suite 输出，以及 15 个 F32/attention 内核经过实际 Buddy 向量化后的第二组主机检查。`verification.json` 和 `manifest.json` 记录相应 LLVM、源码与可执行文件哈希。
- `codegen.json`、`elf-audit.json`：72 个单独 NR ELF 与最终 suite 的完整指令审计、实际编译 flags 和源码哈希。审计成功表示符合已验证 ISA 合约；数值正确性由主机与板上检查分别验证。
- `all/`：最终完整 72 项 FPGA5 验证，全部通过。运行编号 `run-1ee9bd4b8bea48c5`，镜像 SHA256 为 `b6c5f4749239d11cd93d26f6062b75159c87c07184c03599dcc16a990d76dc7f`；包含原始串口日志、上传/DDR 读回结果、最终源码 manifest 和逐 case 检查。
- `aux/`：49 个辅助算子的 FPGA5 串口记录、上传/DDR 读回结果与逐 case 校验。

从本例 `triton/` 目录重新构建并运行：

```bash
# 先激活安装真实 Triton frontend 的 Python 环境。
source ../../common/triton/triton-env.sh
make check GROUP=all JOBS=4
make all GROUP=all JOBS=4
make run GROUP=all FPGA=5
```

公共脚本会打印本次运行的 `run-*` 编号。完成后，将同一镜像的 manifest 和实际运行目录交给公共记录器：

```bash
"$TRITON_PYTHON" ../tools/record_run.py build/suite-all/manifest.json \
  ../../build/fpga-runs/run-实际编号 --output validation/all
```

记录器要求镜像 SHA256 一致、DDR 读回一致、每个 case 恰有一个零错误 PASS、suite 与 NR runtime 正常结束；出现 trap、FAIL 或缺少输出会拒绝记录。远端执行范围保持在用户指定的 `Desktop/fpga-tester-ISCAS` 目录内。
