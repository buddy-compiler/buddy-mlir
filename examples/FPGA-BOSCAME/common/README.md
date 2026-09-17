# 公共平台资源

这些源文件供 `FPGA-BOSCAME` 下的样例复用，不依赖本地参考仓库或临时目录。

| 路径 | 职责 |
| --- | --- |
| `uart/uart.c`、`uart/uart.h` | NR UART：基地址 `0x310b0000`、32 位 MMIO、TX 偏移 `0x20`、固定除数 8 |
| `runtime/crt_uart.S` | 通用单核启动、trap 入口、gp／栈设置和 BSS／TBSS 清零 |
| `runtime/encoding.h` | CRT 所需的 RISC-V CSR 定义 |
| `runtime/bare_runtime.c`、`.h` | C `_init`、输出钩子、堆及可选算子支持 |
| `toolchain.mk` | 可配置的 LLVM 工具选择；从样例 makefile 引入 |

UART 驱动采用 [ModelZoo](https://gitlink.org.cn/michaelcjl/ModelZoo) 的
`thirdparty/nr/src/uart.c` 和 `thirdparty/nr/include/uart.h`。
启动文件和 CSR 定义来自其 `thirdparty/platform-v01/`；C 运行时来自
`examples/tools/bare_runtime.c`、`bare_runtime.h`。
来源提交为 `8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3`，迁入时这六个源文件
均与该快照逐字节一致，原文件中的归属说明予以保留。
CRT 原有注释引用 `riscv-dnn/include/common/crt.S`；不为来源文件补造许可证声明。

`crt_uart.S` 不含固定 UART 或 DDR 地址，通过链接符号设置内存区域。
当前复用方式为 NH 单核入口直接进入 C 运行时，不包含 NR 的 NH→RA 启动调度。
样例需提供链接脚本中的 `_start`、`__global_pointer$`、`__stack_base`、
BSS／TBSS 边界，栈大小与 CRT 的 1 MiB 约定一致。

`hello` 设置 `BARE_RUNTIME_ENABLE_AME=0`，链接时移除未使用的函数和数据。
公共运行时内保留的 AME／RVV 算子代码并不因此获得当前平台的数值验收；
需要 AME 的新样例应按自身 ABI 和硬件单独验证。
