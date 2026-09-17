# Attention PV decode 的代码生成对照

同一 `attention_pv_16x1x128x17/build/kernel.ll`，默认机器优化与同时关闭
machine LICM / machine CSE 的两份汇编保存在本目录。使用相同的
RVV目标、512-bit向量和Buddy batchmatmul优化，linalg/LLVM数值语义没有改变。

完整套件镜像 `9eb6f0850e67040f6e89daddb4f171ef0377f2d3f09c917edfddb91067f048fc`
在真实FPGA5上，该PV decode产生1876/2048处误差，最大绝对误差0.375。
之前默认机器优化版本在板上通过。两份汇编均使用VL16/e32/m1；核心操作均为
load accumulator、17次load B及vfmacc.vf、store accumulator，未出现新的尾部VL。
差异集中在标量循环常量提取、地址计算、GPR分配和栈保存。

因此目前将此现象记录为代码调度敏感行为，尚未确认微架构根因；不能声称是
K尾部向量语义错误，也不能由CPU执行相同LLVM IR通过推断板上执行正确。
本样例对attention恢复经过板测的默认机器优化。只对转置B的FP32线性matmul
使用 `-disable-machine-licm -disable-machine-cse`，用于避免已确证错误的
low-VL whole-register move。两条路径均保留完整ELF ISA/fence审计和数值验证。
