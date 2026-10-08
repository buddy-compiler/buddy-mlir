# MobileNetV3 算子验证记录

2026-10-06 已对 88 个静态 case 分别取得 FPGA PASS；2026-10-08 提交前重跑 88 个已有 Host 验证程序，全部 PASS。BN folding 的 8 个回归测试也全部通过。

**这是不同 bitstream 版本上的逐算子验证，不是完整模型上板结果，也不代表全部 88 个 case 都在 patch6 上通过。**

| 算子 | case 数 | FPGA PASS 版本 |
|---|---:|---|
| residual_add | 4 | nr-patch5 |
| relu | 11 | nr-patch5 |
| hardsigmoid | 7 | nr-patch5 |
| hardswish | 9 | nr-patch5 |
| se_mul | 7 | nr-patch5 |
| mean_hw | 7 | nr-patch5 |
| linear | 1 | nr-patch6 |
| depthwise_conv2d | 9 | nr-patch6 |
| pointwise_conv2d | 32 | nr-patch6 |
| conv_stem | 1 | nr-patch5 |

patch5 使用 14.7456 MHz / UART 分频 8；patch6 使用 11.0592 MHz / UART 分频 6；两者串口均为 115200 8N1，槽位均为 FPGA7。

Linear 在 patch6 原实现出现 3989 处不匹配。修复编译路径后，原 C oracle 的 8 轮、8000 个输出全部通过，最大绝对误差约 8.71e-6；没有放宽容限或修改 oracle。修改为按输出通道向量化，代价是 kernel 内的 64 KiB 权重 tile 和 4 KiB 输入 tile。原始 RVV 归约最小复现仍失败，不能据此声称硬件已修复。

每个成功记录都核对了程序数值 PASS、NR runtime PASS、上传镜像 SHA256 和 DDR 读回。完整 case、版本、run ID、镜像哈希、Host 输出和串口文件哈希见 [summary.json](summary.json)。

保留失败证据：[修复前 Linear](uart/linear-before-patch6.log)、[RVV 归约最小复现](uart/rvv-reduction-minimal-patch6.log)。

patch6 测试复用仓库 runner 的隔离副本，适配差异见 [patch6-runner.patch](patch6-runner.patch)：时钟、交付脚本的 DDR 读回路径和退出等待。未修改公共 runner，也未新增 SSH、串口或上传实现。

本提交不包含模型权重、ELF/BIN、本机依赖或构建缓存。目录中的其他 validation 报告记录各阶段历史状态，当前逐 case 板上结论以本目录为准。

## 全部 case

| Case | 版本 | 原始 UART |
|---|---|---|
| `conv_stem_cin3_cout16_h224_k3_s2_p1` | nr-patch5 | [UART](uart/conv_stem_cin3_cout16_h224_k3_s2_p1.log) |
| `dwconv_c120_h14_k5_s1_p2` | nr-patch6 | [UART](uart/dwconv_c120_h14_k5_s1_p2.log) |
| `dwconv_c144_h14_k5_s1_p2` | nr-patch6 | [UART](uart/dwconv_c144_h14_k5_s1_p2.log) |
| `dwconv_c16_h112_k3_s2_p1` | nr-patch6 | [UART](uart/dwconv_c16_h112_k3_s2_p1.log) |
| `dwconv_c240_h14_k5_s1_p2` | nr-patch6 | [UART](uart/dwconv_c240_h14_k5_s1_p2.log) |
| `dwconv_c288_h14_k5_s2_p2` | nr-patch6 | [UART](uart/dwconv_c288_h14_k5_s2_p2.log) |
| `dwconv_c576_h7_k5_s1_p2` | nr-patch6 | [UART](uart/dwconv_c576_h7_k5_s1_p2.log) |
| `dwconv_c72_h56_k3_s2_p1` | nr-patch6 | [UART](uart/dwconv_c72_h56_k3_s2_p1.log) |
| `dwconv_c88_h28_k3_s1_p1` | nr-patch6 | [UART](uart/dwconv_c88_h28_k3_s1_p1.log) |
| `dwconv_c96_h28_k5_s2_p2` | nr-patch6 | [UART](uart/dwconv_c96_h28_k5_s2_p2.log) |
| `hardsigmoid_count120` | nr-patch5 | [UART](uart/hardsigmoid_count120.log) |
| `hardsigmoid_count144` | nr-patch5 | [UART](uart/hardsigmoid_count144.log) |
| `hardsigmoid_count16` | nr-patch5 | [UART](uart/hardsigmoid_count16.log) |
| `hardsigmoid_count240` | nr-patch5 | [UART](uart/hardsigmoid_count240.log) |
| `hardsigmoid_count288` | nr-patch5 | [UART](uart/hardsigmoid_count288.log) |
| `hardsigmoid_count576` | nr-patch5 | [UART](uart/hardsigmoid_count576.log) |
| `hardsigmoid_count96` | nr-patch5 | [UART](uart/hardsigmoid_count96.log) |
| `hardswish_count1024` | nr-patch5 | [UART](uart/hardswish_count1024.log) |
| `hardswish_count14112` | nr-patch5 | [UART](uart/hardswish_count14112.log) |
| `hardswish_count18816` | nr-patch5 | [UART](uart/hardswish_count18816.log) |
| `hardswish_count200704` | nr-patch5 | [UART](uart/hardswish_count200704.log) |
| `hardswish_count23520` | nr-patch5 | [UART](uart/hardswish_count23520.log) |
| `hardswish_count28224` | nr-patch5 | [UART](uart/hardswish_count28224.log) |
| `hardswish_count47040` | nr-patch5 | [UART](uart/hardswish_count47040.log) |
| `hardswish_count56448` | nr-patch5 | [UART](uart/hardswish_count56448.log) |
| `hardswish_count75264` | nr-patch5 | [UART](uart/hardswish_count75264.log) |
| `linear_m1_n1000_k1024` | nr-patch6 | [UART](uart/linear_m1_n1000_k1024.log) |
| `mean_hw_c120_h14_w14` | nr-patch5 | [UART](uart/mean_hw_c120_h14_w14.log) |
| `mean_hw_c144_h14_w14` | nr-patch5 | [UART](uart/mean_hw_c144_h14_w14.log) |
| `mean_hw_c16_h56_w56` | nr-patch5 | [UART](uart/mean_hw_c16_h56_w56.log) |
| `mean_hw_c240_h14_w14` | nr-patch5 | [UART](uart/mean_hw_c240_h14_w14.log) |
| `mean_hw_c288_h7_w7` | nr-patch5 | [UART](uart/mean_hw_c288_h7_w7.log) |
| `mean_hw_c576_h7_w7` | nr-patch5 | [UART](uart/mean_hw_c576_h7_w7.log) |
| `mean_hw_c96_h14_w14` | nr-patch5 | [UART](uart/mean_hw_c96_h14_w14.log) |
| `pwconv_cin120_cout32_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin120_cout32_h1_w1.log) |
| `pwconv_cin120_cout48_h14_w14` | nr-patch6 | [UART](uart/pwconv_cin120_cout48_h14_w14.log) |
| `pwconv_cin144_cout40_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin144_cout40_h1_w1.log) |
| `pwconv_cin144_cout48_h14_w14` | nr-patch6 | [UART](uart/pwconv_cin144_cout48_h14_w14.log) |
| `pwconv_cin144_cout576_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin144_cout576_h1_w1.log) |
| `pwconv_cin16_cout16_h56_w56` | nr-patch6 | [UART](uart/pwconv_cin16_cout16_h56_w56.log) |
| `pwconv_cin16_cout72_h56_w56` | nr-patch6 | [UART](uart/pwconv_cin16_cout72_h56_w56.log) |
| `pwconv_cin16_cout8_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin16_cout8_h1_w1.log) |
| `pwconv_cin240_cout40_h14_w14` | nr-patch6 | [UART](uart/pwconv_cin240_cout40_h14_w14.log) |
| `pwconv_cin240_cout64_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin240_cout64_h1_w1.log) |
| `pwconv_cin24_cout88_h28_w28` | nr-patch6 | [UART](uart/pwconv_cin24_cout88_h28_w28.log) |
| `pwconv_cin24_cout96_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin24_cout96_h1_w1.log) |
| `pwconv_cin24_cout96_h28_w28` | nr-patch6 | [UART](uart/pwconv_cin24_cout96_h28_w28.log) |
| `pwconv_cin288_cout72_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin288_cout72_h1_w1.log) |
| `pwconv_cin288_cout96_h7_w7` | nr-patch6 | [UART](uart/pwconv_cin288_cout96_h7_w7.log) |
| `pwconv_cin32_cout120_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin32_cout120_h1_w1.log) |
| `pwconv_cin40_cout120_h14_w14` | nr-patch6 | [UART](uart/pwconv_cin40_cout120_h14_w14.log) |
| `pwconv_cin40_cout144_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin40_cout144_h1_w1.log) |
| `pwconv_cin40_cout240_h14_w14` | nr-patch6 | [UART](uart/pwconv_cin40_cout240_h14_w14.log) |
| `pwconv_cin48_cout144_h14_w14` | nr-patch6 | [UART](uart/pwconv_cin48_cout144_h14_w14.log) |
| `pwconv_cin48_cout288_h14_w14` | nr-patch6 | [UART](uart/pwconv_cin48_cout288_h14_w14.log) |
| `pwconv_cin576_cout1024_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin576_cout1024_h1_w1.log) |
| `pwconv_cin576_cout144_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin576_cout144_h1_w1.log) |
| `pwconv_cin576_cout96_h7_w7` | nr-patch6 | [UART](uart/pwconv_cin576_cout96_h7_w7.log) |
| `pwconv_cin64_cout240_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin64_cout240_h1_w1.log) |
| `pwconv_cin72_cout24_h28_w28` | nr-patch6 | [UART](uart/pwconv_cin72_cout24_h28_w28.log) |
| `pwconv_cin72_cout288_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin72_cout288_h1_w1.log) |
| `pwconv_cin88_cout24_h28_w28` | nr-patch6 | [UART](uart/pwconv_cin88_cout24_h28_w28.log) |
| `pwconv_cin8_cout16_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin8_cout16_h1_w1.log) |
| `pwconv_cin96_cout24_h1_w1` | nr-patch6 | [UART](uart/pwconv_cin96_cout24_h1_w1.log) |
| `pwconv_cin96_cout40_h14_w14` | nr-patch6 | [UART](uart/pwconv_cin96_cout40_h14_w14.log) |
| `pwconv_cin96_cout576_h7_w7` | nr-patch6 | [UART](uart/pwconv_cin96_cout576_h7_w7.log) |
| `relu_count144` | nr-patch5 | [UART](uart/relu_count144.log) |
| `relu_count225792` | nr-patch5 | [UART](uart/relu_count225792.log) |
| `relu_count24` | nr-patch5 | [UART](uart/relu_count24.log) |
| `relu_count32` | nr-patch5 | [UART](uart/relu_count32.log) |
| `relu_count40` | nr-patch5 | [UART](uart/relu_count40.log) |
| `relu_count50176` | nr-patch5 | [UART](uart/relu_count50176.log) |
| `relu_count56448` | nr-patch5 | [UART](uart/relu_count56448.log) |
| `relu_count64` | nr-patch5 | [UART](uart/relu_count64.log) |
| `relu_count68992` | nr-patch5 | [UART](uart/relu_count68992.log) |
| `relu_count72` | nr-patch5 | [UART](uart/relu_count72.log) |
| `relu_count8` | nr-patch5 | [UART](uart/relu_count8.log) |
| `add_count18816` | nr-patch5 | [UART](uart/add_count18816.log) |
| `add_count4704` | nr-patch5 | [UART](uart/add_count4704.log) |
| `add_count7840` | nr-patch5 | [UART](uart/add_count7840.log) |
| `add_count9408` | nr-patch5 | [UART](uart/add_count9408.log) |
| `se_mul_c120_h14_w14` | nr-patch5 | [UART](uart/se_mul_c120_h14_w14.log) |
| `se_mul_c144_h14_w14` | nr-patch5 | [UART](uart/se_mul_c144_h14_w14.log) |
| `se_mul_c16_h56_w56` | nr-patch5 | [UART](uart/se_mul_c16_h56_w56.log) |
| `se_mul_c240_h14_w14` | nr-patch5 | [UART](uart/se_mul_c240_h14_w14.log) |
| `se_mul_c288_h7_w7` | nr-patch5 | [UART](uart/se_mul_c288_h7_w7.log) |
| `se_mul_c576_h7_w7` | nr-patch5 | [UART](uart/se_mul_c576_h7_w7.log) |
| `se_mul_c96_h14_w14` | nr-patch5 | [UART](uart/se_mul_c96_h14_w14.log) |
