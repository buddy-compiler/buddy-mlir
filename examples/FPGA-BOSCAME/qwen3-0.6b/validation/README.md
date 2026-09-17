# FPGA 算子验证记录

板卡：FPGA5，NR / RAV0.5，UART115200，系统时钟14.7456MHz。
运行工具为公共 `fpga_run.sh`，远端所有执行均在用户指定的
`Desktop/fpga-tester-ISCAS` 下；上传文件放在该目录内独立 `fpga-runs/run-*`。

## 最终 linalg 套件

[all/verification.json](all/verification.json) 记录完整 **72/72 PASS**，包括
12个AME整数矩阵、11个RVV FP32线性算子、4个RVV attention和45个辅助算子。
对应 `run-a2db3e72d49d4359`，镜像SHA256：
`0fc59316d0bb0af07174936d20d792988cb1eea6e21babcd3d5b123f9efc1cce`。
所有输出均与独立参考匹配，DDR读回一致，suite与NR运行时均返回PASS。
此前分组记录 `ame/`、`aux/` 保留作开发过程证据；当前完整结果以 `all/` 为准。

Triton前端的对应记录位于 [triton/validation](../triton/validation)。
排查过程和硬件限制见 [debugging.md](debugging.md)，失败探针不计入算子通过数。

## 记录内容

每组记录包含：

- `manifest.json`：镜像SHA256、实际包含的case、linalg源码哈希。
- `uart.raw.log`：真实串口输出。
- `result.json`：上传/DDR读回一致性、完成标记与运行状态。
- `verification.json`：逐case成功标记检查，以及原始日志哈希。

重跑后可记录结果（从本例目录执行）：

```bash
python3 tools/record_run.py build/suite-aux/manifest.json \
  ../build/fpga-runs/run-实际编号 --output validation/aux
```

记录器要求镜像哈希一致、DDR读回正确、每个case恰有一个零错误PASS标记、
集合与NR runtime完成且没有FAIL/trap。仅看到启动文字不会算成功。

测试输入为可复现合成数据。它验证算子、ABI、代码生成、启动和硬件执行，
不代表已经验证官方checkpoint、整模型文本质量或所有动态序列长度。

`rvv-capabilities/` 记录了 FPGA5 上23项 e32/m1、VL16 的RVV能力检查，
全部 `SUPPORTED numeric=PASS`，以及 `nr_copy_bytes` 的384项长度/对齐检查。
这提供扩展本目录 RVV 指令路径的硬件依据，不能替代每个新生成kernel的数值测试。
`rvv-capabilities-expanded/` 保存后续33项探针的原始串口日志、上传结果及逐项分类。
对应 `run-6e31d5d081284634`，镜像 SHA256 为
`6cad81a06f303d5636fee2f3572821793e0adc25c747964f423fbbe4e9025808`。
29项数值通过，包括 `vfmadd.vv`、e32/m1/VL16下的 `vmv1r.v`、`vsetivli`、e8/mf4/VL16 读写、
e64/m1/VL2和VL4读写。`vl1re32.v + vs1r.v` 无trap但有12处数值错误；
读取 `vlenb`、`vl`、`vtype` CSR均触发illegal instruction。因此生产编译继续拒绝
whole-register memory及这些CSR访问。

`rvv-capabilities-low-vl/` 保存34项探针 `run-baf62e6e7b5242a1`，镜像SHA256
`f07bb95c4cd904a4e22fd1d86e2fd60d01b1f13cfb890f0a78f4ef6e88760fc1`。
新增测试先加载16个不同的f32源/目标元素，再设e8/m1/VL1执行 `vmv1r.v`，最后
切回e32/m1/VL16读回逐位比较，16个元素全部错误。原e32/m1/VL16测试仍通过。
这说明该板的whole-register move不能满足与VL/VTYPE无关的标准语义，生产路径
因此**禁止全部whole-register move**，不能从单一配置的通过推断通用支持。
LLVM生成FP32代码时同时使用 `-disable-machine-licm -disable-machine-cse`，
使每个dot累加器直接在e32/VL16下清零，避免生成该copy；最终ELF再次检查。
本次探针29项通过、3项illegal、2项数值失败，整体FAIL符合预期，copy384项通过。

该扩展探针的预期整体状态是 **FAIL**：数值探测失败使 `launch` 返回1，公共脚本
正确记录 `status=ERROR`，不能将其中的“probe completion: PASS”误当作全部能力通过。
DDR读回一致，探针完成，后续384项copy校验通过。这是硬件能力的负面证据，
与模型算子验证失败不同，也不应使用逐算子 `record_run.py` 将它标为成功。

`compiler-tests.log` 保存35项NR/GEM5相关编译器回归。
`host-check.log` 保存72个原始linalg算子的独立CPU数值检查，以及15个FP32矩阵/
attention算子经过实际Buddy vector pass后的第二组CPU数值检查。
`launcher-isa-tests.log` 保存47项上传/重连、AME编码和完整ELF指令检查回归。
`rvv-machine-code.json` 保存这15个FP32样例最终ELF/bin哈希与实际RVV/AME指令计数，
包括已链接运行时的指令；全部通过完整ELF审计，无whole-register move/memory。
该文件只证明生成代码及ISA审计结果，上板结果以对应串口运行记录为准。
