# NR NH/RA 算子运行时

此目录服务当前 NR FPGA 的独立算子与算子集合。NH 从 `0x80000000` 启动，
初始化公共 `../uart/uart.h` 驱动，然后启动 RA。RA 开启 FS/VS/XS/AME MS，
清除普通 BSS，并调用样例提供的 `int launch(void)`。返回 `0` 表示校验成功；
其他值、trap、heap 耗尽均传播为 UART 上的 `FAIL`。hello 继续使用原来的单核运行时。

RA 日志写入独立的 64 KiB DDR 环形缓冲，NH 转发到 UART；满环时 RA 等待 NH
消费，支持累计输出超过 64 KiB。`nr_write(bytes, length)` 可输出包含 NUL 的原始
字节；`nr_puts` / `nr_hex32` / `nr_hex64` 和兼容的 `print_uart*` 也可在 launch
中使用。数值输出是十六进制，`nr_cycles()` 返回 RA 周期计数。

`nr_getchar()` 在 RA 从 NH 提供的 4 KiB 输入环读取一个字节，没有输入时返回 -1。
只有 NH 访问 UART MMIO 和执行缓存维护。每个核写的游标独占缓存行；NH 写入后
flush，读取 RA 所有的数据前 invalidate。共享控制区是 NOLOAD，启动时由 NH
显式初始化。输入环满时丢弃后续字节；发送者应等待提示符，遵守应用输入长度
上限。固件能打印提示符、主机 write 成功均不能代替实际 RX 验证。

## 构建接入

从算子目录使用相对路径包含公共工具链和此 make 片段，例如：

```make
include ../../common/toolchain.mk
include ../../common/nr/nr.mk

build/launch.o: launch.c | build
	$(RISCV_CC) $(NR_CFLAGS) -c $< -o $@

build/operator.elf: build/launch.o build/operator.o $(NR_OBJECTS) $(NR_LINKER_SCRIPT)
	$(RISCV_LD) -T $(NR_LINKER_SCRIPT) --gc-sections -o $@ $(filter %.o,$^)

build/operator.bin: build/operator.elf
	python3 $(NR_DIR)/../../tools/check_nr_elf.py $< --objdump "$(RISCV_OBJDUMP)" --output build/elf-audit.json
	$(RISCV_OBJCOPY) -O binary $< $@
```

相对路径按使用位置调整。launch 包含 `nr_runtime.h`；不要同时链接
`common/runtime/bare_runtime.c` 或 `common/uart/uart.c`，因为这里提供了对应
内存、日志和启动符号。算子对象应使用与当前 FPGA 匹配的 BOSCAME lowering；
通用 C 运行时和 launch 必须用 `-march=rv64gc_zicbom`，不启用 V 扩展。
仅关闭自动向量化不足以阻止LLVM为ABI描述符或寄存器保存生成vector CSR读取及
whole-register访存；这些形式在当前NR板上不支持。只有经过审计的算子汇编及
`nr_copy.S` 单独使用含V的目标选项。

生成二进制前使用公共 `tools/check_nr_elf.py` 审计完整ELF的所有可执行字节，
覆盖算子、launch和运行时；不能只检查算子汇编。检查拒绝未验证的AME/RVV、
vector CSR与whole-register move/memory，并确认AME及vector访存两侧的fence。
qwen3样例的单算子与集合构建均已接入此步骤，报告保存在 `elf-audit.json`。
能力探针有意包含不支持指令，故不使用该生产审计。

`ame_sync.c` 提供 ModelZoo NR 的 `ame_fence()`：它执行带 fence 的 1×1 AME
同步操作，使用普通 B load，避开 NR 不支持的转置形式。此操作会改变 AME 的
配置及寄存器；只在算子边界调用，不能在尚需保留 tile 寄存器时调用。生成的
算子自身也必须按照 NR 合约在每条 AME 指令前后插入 `fence rw, rw`。

## 大数组与内存约定

普通 BSS 由 RA 清零，NH 不触碰，以免留下不一致的脏缓存行。NH/RA 分别有
1 MiB 栈。固定信号 mailbox 是 `0x80010000`。代码、BSS、栈和简单 bump heap
放在低 DDR，范围不超过 `0xb0000000`。

ModelZoo 记录 RA 在 `[0xb0000000, 0xb8000000)` 的访问故障，因此大 workspace
从 `0xb8000000` 开始。声明方式：

```c
#include "nr_runtime.h"
static unsigned char arena[640u * 1024u * 1024u] NR_WORKSPACE;
```

对于数百 MiB 的 arena，Clang 的自定义 C section 在中间 `.o` 中可能生成
零填充 PROGBITS。可使用与 qwen3 样例相同的汇编声明，直接让中间对象也是
NOBITS，避免大对象文件及旧版 LLD 的 section-type warning：

```c
extern unsigned char arena[640u * 1024u * 1024u];
__asm__(".pushsection .workspace,\"aw\",@nobits\n"
        ".balign 64\n.globl arena\narena:\n"
        ".zero 640 * 1024 * 1024\n.popsection\n");
```

`.workspace` 为 `NOLOAD`：不增加上传的 bin 大小，也不在启动时清零。
launch 必须初始化算子实际会读取的每个元素；测试 embedding gather 时可以只
初始化确定会被索引的行。链接符号 `__workspace_start`、`__workspace_end`
描述此段边界。链接器校验它不超过 `0x100000000`，不允许穿过故障区。

`malloc` / `aligned_alloc` 使用低 DDR bump heap，`free` 不回收；耗尽会打印
请求大小、当前游标和上界并停止。重复调用编译图时，在调用前保存
`uintptr_t mark = nr_heap_mark()`，将所有仍需使用的返回 tensor/KV 复制到持久
buffer、完成异步访问后调用 `nr_heap_reset(mark)`。reset 后禁止继续使用该次
调用分配的任何指针；mark 之前的分配保留。集合运行也可复用 `.workspace`
中的 arena。另提供标量 `memcpy` / `memmove` /
`memset` / `memcmp` 和 MLIR `memrefCopy` ABI（rank 最大为 8）。

`nr_copy_bytes(dst, src, bytes)` 用已在 ModelZoo NR 上验证的 RVV copy 加速互不
重叠的连续内存复制：e32/m1、每次至多 16 个元素、向量访存两侧 fence。源和目标
都按 4 bytes 对齐时走向量路径；其他对齐以及最后 0～3 bytes 用标量复制。
它会覆盖 v8、VL、VTYPE，调用方不得跨调用保留向量状态。边界校验和更多 RVV
能力调查在 [probes](probes/README.md)。

## 数学函数与来源

启动、mailbox、日志同步和 AME 同步协议依据 ModelZoo 的
`thirdparty/nr`、`examples/buddy-qwen35-fpga/platform/nr` 和
`examples/tools/bare_runtime.c` 迁移、精简。`nr_math.c` 的 exp/log 表及
`expf`、`logf`、`powf`、`tanhf`、`erff` 迁移自
`examples/buddy-qwen35-fpga/runtime/src/qwen35_bare_math.c`；原文件注明其
Arm exp/log 表来自 MIT licensed optimized-routines。

参考仓库：https://gitlink.org.cn/michaelcjl/ModelZoo.git ，提交
`8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3`。构建不依赖该仓库。

新增 `sqrtf` 使用 `fsqrt.s`。`sinf` / `cosf` 面向 RoPE，采用 double 角度约简与
多项式，支持有限输入 `|x| <= 2^20` 弧度；超出范围返回 NaN。它们以及原来的
pow/tanh/erf 是 freestanding 样例辅助实现，不是完整 IEEE libm。
