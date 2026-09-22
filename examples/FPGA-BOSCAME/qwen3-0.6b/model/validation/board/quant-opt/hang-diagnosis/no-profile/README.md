# 无 profiler 复跑：RA trap，未完成验收

2026-09-18，FPGA5 `run-2755b33255ca4832`。
配置为 28 层、max sequence length 128、16-token prefill + 8-step decode，
固定原始文本 `What is France?`。本次未启用 kernel profiler、progress/probe、
intermediate instrumentation；保留原验收镜像的阶段输出和独立数值比较。

## 结果

- runner `ERROR`，`RA returned: FAIL`，不是正常完成或采集超时。
- boot image、权重、tokenizer 三段装载回读全部一致。
- 板上 tokenizer PASS；prefill 和 decode position 16、17 完整完成，
  对应 token 为 49000、374、264，完整 logits 和有效 K/V 比较通过。
- decode position 18 已返回模型图、执行图末同步、选择 token 并保留 K/V，
  logits 和 key_cache 比较均零误差；随后发生 trap，value_cache 没有比较结果。
- 因此仅完成 2/8 个完整 decode 阶段，不能记录为模型全流程 PASS。

```text
[compare] logits position=00000012 count=00025180 max_abs_bits=00000000 mean_abs_bits=00000000 PASS
[compare] key_cache position=00000012 count=00085000 max_abs_bits=00000000 mean_abs_bits=00000000 PASS
[nr] RA TRAP mcause=0x0000000000000018 mepc=0x0000000080167710 mtval=0x0000000080167710
[nr] RA returned: FAIL
```

## 异常位置与判断范围

精确 ELF 的 `collect` 函数范围为 `0x8016721a..0x801677e4`，包含内联的
`compare_reference`。`0x8016770c` 调用 `report_error` 报告 key_cache，
返回地址 `0x80167710` 的 ELF 指令为 `4581` / `li a1, 0`；之后才进入
value_cache 比较循环。见 [trap-disassembly.txt](trap-disassembly.txt)。

这次异常现场属于图计算后的校验路径，不能描述为仍在等待某个 AME kernel。
ELF 指令只说明本地镜像内容，尚未读取故障时 DDR/指令缓存内容。
也不能仅凭普通指令及 `mepc == mtval` 断定是取指、缓存或硬件故障。

`common/nr/crt.S` 的 RA trap entry 直接读取 mcause/mepc/mtval，
`nr_runtime.c::nr_ra_trap` 按对应参数打印后失败退出。两份 v0.5 文档没有
异常编号表，尚无法确认 E6 RA 对异常编号 24 的定义；不能将其当作标准
编号 1 的 instruction access fault。硬件文档对应的设计版本是
`nanhu-fpga-aps.git` 的 `feat/AME` commit
`6c4d6dc1ab8197cac1a7ef2cb0e8a61fb9513402`；后续可核对该版本 E6 异常定义。

本次未填充/填充后的 boot bin、权重和 tokenizer 都与此前成功的
`run-c9c51a235a154732` 使用的本地基线逐字节相同；hash 见
[summary.json](summary.json)。当前证据说明去掉 profiler 仍可能异常，
不能把 profiler 作为所有故障的唯一原因；本次 trap 与此前无输出停滞是否同因仍未知。

## 已完成阶段的时间

配置时钟 14,745,600 Hz，经本次 UVHS 日志核实。

| 阶段 | 图计算秒数 | 含准备、选择、KV 保留的秒数 |
| --- | ---: | ---: |
| prefill | 247.921177 | 252.458504 |
| decode position 16 | 67.407700 | 70.837485 |
| decode position 17 | 67.610121 | 71.040061 |

没有第三步的完整计时行，不能报告 8 步 decode 平均值。

## 核验与证据

产物目录：`model/build/quant-opt/no-profile-rerun-28l-cap128/`。
原始日志：`examples/FPGA-BOSCAME/build/fpga-runs/run-2755b33255ca4832/`。
本目录保留日志、装载 manifest、ELF audit、构建配置、反汇编及失败的数值检查结果。
没有启动下一次硬件任务。

在仓库根目录可重查异常地址：

```bash
MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
IMAGE="$MODEL/build/quant-opt/no-profile-rerun-28l-cap128/image"
llvm/build-2d26/bin/llvm-objdump -d \
  --start-address=0x801676b0 --stop-address=0x80167780 "$IMAGE/qwen_model.elf"
```
