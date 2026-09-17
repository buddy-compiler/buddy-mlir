# 已暂停：2026-09-17

按用户“到最近检查点停下”的要求暂停；未启动28层测试，等待用户通知。
预置原始UTF-8提示词为 `What is France?`。模板、tokenizer、模型计算、token选择和
增量文本解码均在FPGA；UART RX暂缓。

## 本轮确认通过

- **Stage B单层中间值**：`run-550440b96df24b2c`，16-token prefill一次、连续8次decode，
  **414项选定完整中间张量比较、27项完整末位logits/有效KV比较均max/mean error=0**。
  板上模板/编码/文本解码通过；全部装载段readback匹配；严格归档通过。
  证据：`validation/board/review/model-1l-intermediates-no-profile/`。
  该镜像关闭kernel profiler，保留中间值比较和probe进度；耗时包含诊断，不能作吞吐。
  覆盖限制：K RoPE通过KV、SwiGLU通过down输入、final norm只检查最后一行；不含int32
  accumulator及所有未列出的隐藏张量。
- **四层固定文本诊断版**：`run-b8d3ed3ead854a24`，16+8，27项完整logits/KV零误差、
  tokenizer/文本解码和profile检查通过。证据：
  `validation/board/review/model-4l-native-fixed-text-progress/`。
  逐kernel UART计入graph时间，不能用作纯推理速度；四层输出不代表完整模型质量。
- 本地31项回归通过：`validation/board/review/checkpoint-tests.json`。
- 此前native-K 1/4/28层host和1层FPGA证据继续有效；不能向新镜像或28层板测外推。

## 已修复与未解决必须区分

1. **已确认并规避旧链接器缺陷**：PATH中的Ascend LLD15在StageB镜像中将wrapped函数
   符号放到函数中间，跳过保存栈/ra。相同objects用LLD20或旧LLD关闭relaxation恢复入口。
   builder现要求LLD>=20，记录路径/版本/二进制hash；保留multicall symlink调用名。
   新LLD20镜像完整单层测试通过。证据：`validation/board/review/linker-wrap-audit/`。
2. **已修正asm合约**：`common/nr/ame_sync.c`中msettype指令与GPR clobber合并到同一asm。
   修正已用于通过的单层镜像，但不能说它解释了四层停顿。
3. **kernel profiler相关停顿仍未根治**：两次StageB带profiler版本在46项prefill中间值
   全通过后停在profile输出途中；关闭profiler的隔离版本完整通过。移除profiler同时改变
   wrapper/fence/布局/时序，不能仅凭这个实验断言是UART或某一条指令的问题。
4. **四层无逐kernel日志版仍未验收**：旧版本1200秒超时；LLD20+asm修正版仍只有535字节
   tokenizer PASS/prefill begin，后被主动停止。它仍启用kernel profiler，只是不输出逐
   kernel begin/end。证据：`validation/board/review/model-4l-native-syncfix-stopped/`。
   4层新旧ELF所有JAL/JALR/branch/ret一致，未发现1层那种入口错位，见
   `validation/review-linker-4l-static.json`。不能归因“只慢”或“只卡平台”。

## 用户通知继续之后

先做四层完全关闭kernel profiler的固定文本16+8数值对照，以区分profiler副作用；
必要时读取RA/NH状态与console生产/消费计数。已有UART字节数不等于RA已生产字节数，
不能据此排除backpressure。修复前不要把profiler镜像当作稳定执行路径。

随后再做28层固定文本16+8；完整中间值/数值/文本/性能记录和最终构建说明仍需完成。
准备好的28层LLD20+asm修正版位于 `build/review-native-28l/`，均未上板；HIGH占用
791.30MiB/1152MiB，动态heap峰值尚未实测。

`examples/BuddyQwen3`未改。用户随后要求提交此检查点及依赖版本；本次提交包含此前的
staged算子工作与本检查点的模型工作。远端FPGA操作保持限定目录，未继续模型测试。
