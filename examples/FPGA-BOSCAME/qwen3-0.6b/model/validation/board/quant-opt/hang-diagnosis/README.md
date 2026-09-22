# 28-layer profile 停滞定位

配置：28 层、capacity 128、16-token prefill、8-step decode，固定文本
`What is France?`。这是一组故障诊断，不是完整数值验收或性能结果。

## resync 诊断镜像

`run-1daeec434e5b4b52`，镜像 SHA256：
`2bb1e3ff1c99903610d4aba6a4d90a88b671f02d469a317d8d9ee81c2ff91a6e`。
保留原 profiler 的每调用 `ame_fence()`，只增加 kernel 返回、图执行和
收集结果的阶段标记。原图、adapters、公共 runtime 对象与成功基线逐字节一致。

实测：

- prefill 的 873 次 kernel 调用全部完成，logits、K/V 对照通过。
- decode position=16 完成 454 次调用，随后进入第 15 层（零基 layer 14）
  `qwen_graph_w8a8_mm_101_matmul_1x1024x2048`，没有看到 kernel 返回标记。
- 前一个 `quantize_1x2048` 的 returned/end 都已输出，因此它及其附加同步完成。
- 本地与远端 UART 均为 333773 bytes，最后一行是该 matmul 的 begin。
  `resync/stall-first.json` 和 `stall-final.json` 的采样相隔 61.77 秒，
  日志 SHA256 完全相同；此前 10:35:41 UTC 的观察也已是同一末行/大小。
- 手动终止本次专属 worker，结果为 `INTERRUPTED`，三个 DDR segment 的
  readback 全部匹配。停止后确认 worker 退出、平台锁释放、FPGA5 UART 无占用。

这把停滞区间缩到 matmul 调用及返回标记之前，未能定位具体 PC。
该调用包含 16 个 grid，每 grid 固定 32 次 K=64 的 AME 计算及普通内存访问；
不存在模型数据控制的软件等待循环。尚不能区分具体 AME 指令或访存停滞，
也不能仅据此证明或排除之前附加 resync 的影响。

## 静态排查

成功与失败镜像的 graph、adapters、NR runtime、copy/math、tokenizer 和
AME sync 对象一致。高 DDR 权重/KV/tokenizer/workspace 分区一致、无重叠。
heap 容量约 738.9 MiB，而已完成 decode 的峰值为 87.37 MiB，且没有逐步增长。
RA 栈为 1 MiB，主要固定调用帧约 278 KiB。没有发现明显容量或布局错误；
这些检查不能排除运行时坏指针或硬件访问异常。

## 对照实验

保留相同诊断标记、生产 kernel 和图末公共同步，仅使用
`--profile-sync=fence` 将 profiler 附加的 dummy AME resync 改为
`fence rw,rw`。默认模式仍为 `ame-resync`。

该对照 `run-7c801b4cccf3442e` 也完成 prefill，最终 UART 停在同一个
decode position=16、layer 14 的 matmul begin（333773 bytes），随后被手动终止。
其观察窗口较短，未保存与 resync 对照相同的长时间不变快照；不能声称证明了
可重复的硬件死锁。移除 profiler dummy resync 没有让此诊断运行完成。

## 无插桩复跑

用户随后要求移除 profiler 复跑。`run-2755b33255ca4832` 使用与此前成功
基线相同的 boot bin、权重及 tokenizer，完成 prefill 和前两步 decode；
第三步 position=18 在图计算后的 K/V 数值校验路径报 RA trap，
`mcause=0x18`、`mepc=mtval=0x80167710`，runner ERROR。
去掉 profiler 仍可能异常，本次 trap 与之前无输出停滞是否同因尚未确定。
详见 [无插桩证据](no-profile/README.md)。
