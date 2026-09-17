# NR RVV 能力探针

```sh
make -C examples/FPGA-BOSCAME/common/nr/probes
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/common/nr/probes/build/rvv_probe.bin --fpga=5
```

`rvv_probe.bin` 用已在 ModelZoo NR 验证的 `vsetvli` 设置 e32/m1，请求 VL=16，
逐项检查浮点加减乘除、FMA、sqrt、sum/max reduction、min/max、int/float conversion
和 broadcast。输入包含正负有限数，输出附有越界 canary。每个测试打印实际 VL。
扩展测试还包含 `vfmadd.vv`、`vmv1r.v`、whole-register memory、`vsetivli`、
只读vector CSR，以及e8/mf4/VL16、e64/m1/VL2和VL4的内存操作。移动/复制结果
逐位比较；各项能力只在对应VL、SEW、LMUL与测试输入下成立。

此镜像专门探测生产白名单以外的指令，因此直接组装 `rvv_ops.S`；它不扩展生产
`nr_isa.py` 的名单。探针临时替换 RA 的 `mtvec`，只对 illegal instruction
（mcause=2）记录 PC/mtval 并跳过该条 32 位指令；其他异常仍交给终端 trap handler。
完成后恢复原 `mtvec`。

- `SUPPORTED numeric=PASS`：本测试输入和配置下执行并通过数值检查。
- `UNSUPPORTED`：该测试序列触发 illegal instruction；PC/mtval 可定位具体指令。
- `EXECUTED NUMERIC_FAIL`：没有 illegal trap，但结果或 canary 不符。

`verify RVV probe completion: PASS` 只表示能力探针跑完，**不表示全部指令受支持**。
UNSUPPORTED 本身不使进程失败；数值失败或后续 copy 校验失败返回非零。
这不是完整 RVV 合规测试，不覆盖所有舍入模式、NaN/Inf/subnormal、
VL/LMUL 组合或与 AME 交错运行；引入生产 kernel 后仍需要对应算子完整数值验证。

另一个镜像 `build/copy_probe.bin` 验证公共 `nr_copy_bytes`：24 种长度（0 到 513
bytes）×16 种源/目标对齐组合，共 384 项，检查整个输出缓冲及两侧 canary。
此镜像使用普通终端 trap 行为，退出/校验失败不能当作通过。

FPGA5已记录的33项运行结果见
[`qwen3-0.6b/validation/rvv-capabilities-expanded`](../../../qwen3-0.6b/validation/rvv-capabilities-expanded)。
29项通过；whole-register load/store序列有12处错误，3项vector CSR读取为illegal。
因此该能力探测镜像整体返回FAIL是预期结果，生产路径仍拒绝这些失败的形式。
后续34项探针新增 `vmv1r.v` 在VL1/e8后的使用，16个f32元素全部错误；
见 `qwen3-0.6b/validation/rvv-capabilities-low-vl`。生产路径因此也禁止所有
whole-register move，不把e32/VL16的通过泛化到其他配置。

UART 和 DDR 小探针可独立构建，不需要模型权重：

```sh
make -C examples/FPGA-BOSCAME/common/nr/probes \
  build/uart_rx_probe.bin build/console_wrap_probe.bin build/ddr_probe.bin
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/common/nr/probes/build/uart_rx_probe.bin --fpga=5 \
  --capture-seconds=90 --completion-marker='[nr] RA returned:' \
  --uart-send=$'BOSC NR UART RX\n' --uart-send-delay=5
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/common/nr/probes/build/console_wrap_probe.bin --fpga=5 \
  --capture-seconds=45 --completion-marker='[nr] RA returned:'
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/common/nr/probes/build/ddr_probe.bin --fpga=5 \
  --capture-seconds=30 --completion-marker='[nr] RA returned:'
```

RX 探针必须逐字节收到指定的 16-byte 行并返回 PASS；只看到 ready 或主机 write
成功不算通过。TX 探针输出 1,024 行固定模式，共 67,584 bytes，回收 UART 日志后
需逐字节比较 BEGIN/END 之间内容，验证跨越 64 KiB 环形缓冲边界。

DDR 探针会覆盖 `0xb8000000` 起每 64 MiB 一条缓存行，直到 `0xfc000000`，
另测 `0xfffff000`，共 19 条、每条 64 bytes。全部写完后再全部比较，检查采样地址
之间的别名。它仅证明这些地址上的访问，不代表完整 DDR 容量或可靠性测试；
必须作为独立镜像执行，不能与仍需保留的模型权重/数据同时使用。

本次复核的原始板上证据在
[`model/validation/board/review`](../../../qwen3-0.6b/model/validation/board/review)。
