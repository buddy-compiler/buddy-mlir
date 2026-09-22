# FPGA-BOSCAME 裸机样例与公共资源

面向 NR／RAV0.5 FPGA 平台。当前 `hello` 由 NH 单核运行，适合先验证
编译、DDR 加载和 UART 输出。其他样例可复用下列驱动、运行时及上板工具。

```text
FPGA-BOSCAME/
├── common/
│   ├── toolchain.mk       # LLVM 工具选择，可通过 make 参数覆盖
│   ├── uart/              # NR UART 驱动
│   ├── runtime/           # hello 使用的 NH 单核运行时
│   ├── nr/                # 算子使用的 NH→RA、AME/RVV 运行时
│   └── triton/            # Triton 编译器安装、版本与兼容补丁
├── fpga_run.sh            # 公共上板入口
├── tools/                 # SSH 上传、远端执行、离线测试
├── qwen3-0.6b/            # linalg 算子；triton/ 提供对应前端
└── hello/
    ├── hello.c
    ├── makefile
    ├── platform/hello_nr.ld
    └── tools/host_check.c
```

## 构建 hello

从仓库根目录执行：

```bash
make -C examples/FPGA-BOSCAME/hello all check size
```

工具链配置在 [`common/toolchain.mk`](common/toolchain.mk)。默认按顺序寻找
仓库的 `llvm/build-2d26/bin`、`llvm/build/bin` 中的 clang，再回退到 `PATH`。
在所选目录内找不到的工具（例如 `ld.lld`）也从 `PATH` 获取。
需要 clang（支持 RV64）、LLD、llvm-objcopy；反汇编使用 llvm-objdump。
无需在源码中填写安装目录，配置方式为：

```bash
# LLVM_BIN 相对于 make -C 指定的样例目录；也接受用户提供的绝对路径。
make -C examples/FPGA-BOSCAME/hello LLVM_BIN=../../../llvm/build/bin all

# 将所需工具加入 PATH 后，可只使用 PATH 中的工具。
make -C examples/FPGA-BOSCAME/hello LLVM_BIN= all
```

也可分别设置 `RISCV_CC`、`RISCV_LD`、`RISCV_OBJCOPY`、`RISCV_OBJDUMP`。
切换工具链或编译选项后先执行 `make clean` 再构建。
这些源文件已收进当前目录，构建无需额外克隆 ModelZoo。

## 构建 Qwen3 none 整模型

完整构建命令见 [Qwen3 README：构建 none 整模型二进制](qwen3-0.6b/README.md#构建-none-整模型二进制)，
包括从零创建 Python 环境、编译 LLVM/Buddy/Triton、下载资源、构建图/kernel/权重和
ELF/BIN、准备三段 DDR 文件，不要求已有任何 `build/`。关键参数为 `--ame-startup=none --graph-sync=ame-resync
--ame-cache-sync=none --console-mode=append-only --console-capacity=4194304
--console-drain=live`，不启用额外诊断选项。

重新构建输出到新目录，必须重新上板验收；下面的复跑命令则只使用原始已验收镜像。

## 上传并运行：Qwen3 原始 none 镜像

Qwen3 整模上板使用已经验收的 `production-control` 原始镜像，
构建配置为 `--ame-startup=none`：完整 28 层、最大序列长度 128、
16-token prefill 和 8 步 decode。`none` 仅表示不额外执行首次图调用前的
启动同步；图结束后的 `ame_fence()` 和 kernel 内部同步仍然保留。

最近一次复跑 `run-1f8b9ec0c3894135` 完成 27 项 logits/KV 校验、固定文本校验和
`RA returned: PASS`，严格验收为 `MODEL_RUN_NUMERIC_PASS`。证据见
[none 验收归档](qwen3-0.6b/model/validation/board/startup-ab-20260922/none-run-1f8b9ec0c3894135/verification.json)。
这是多次成功的基线，不代表任意重新构建的同名配置已经通过验收。

从仓库根目录执行；直接复用下列产物，不重新编译：

```bash
set -euo pipefail
QWEN_MODEL=examples/FPGA-BOSCAME/qwen3-0.6b/model
QWEN_NONE="$QWEN_MODEL/build/console-fix-20260921/production-control"

# 核对已验收 ELF 和实际上板的补齐镜像，防止误用其他构建。
printf '%s  %s\n' \
  9a604617c171be8577dcb5aa6cffe0e05cee2ed549f1f27f55b8af444dfc9f75 "$QWEN_NONE/image/qwen_model.elf" \
  f4a23079f4272aa4853ec33ccf4f8d0f3c5b5884a0041cea6dc57f92ae088742 "$QWEN_NONE/prepared/image.bin" \
  | sha256sum --check

"$QWEN_MODEL/tools/run_model.sh" "$QWEN_NONE/prepared" \
  --fpga=5 \
  --remote-dir=/home/hjuser/Desktop/fpga-tester-ISCAS \
  --capture-seconds=1800 --startup-timeout=900
```

`run_model.sh` 自动装载 `prepared/` 中的 boot image、权重和 tokenizer 三段，
使用 `ddr-load.plan` 检查大小、SHA256 和 DDR 回读，并设置模型完成标记。
不要把模型 `image.bin` 当成 hello 单段镜像直接上传；缺少权重或 tokenizer
不构成同一实验。完整通过还应看到 27 项 `[compare] ... PASS`、
`verify fixed text validation: PASS` 和 `[nr] RA returned: PASS`。

这些 `build/` 产物不提交 Git；新 checkout 不能仅靠上述命令运行。
若产物缺失，应先恢复已验收产物，或按[从零构建说明](qwen3-0.6b/README.md#构建-none-整模型二进制)
重建并重新验收。不要用另一个镜像冒充此处指定 SHA256 的基线。

脚本使用 SSH 别名 `fpga`。上述命令显式限定服务器工作目录为
`/home/hjuser/Desktop/fpga-tester-ISCAS`，在该目录内调用 `make uv_run5`。
SSH 登录需使用免交互密钥认证，服务器需已有 UVHS 环境和配套平台 Makefile。

同一目录只能有一个 UART worker。SSH 中断时使用本次实际 run ID 恢复，
不重新运行上传命令、不打开 minicom：

```bash
QWEN_RUN_ID=run-0123456789abcdef  # 替换为本次任务的实际 ID
examples/FPGA-BOSCAME/fpga_run.sh "$QWEN_NONE/prepared/image.bin" \
  --fpga=5 --resume-run="$QWEN_RUN_ID" \
  --remote-dir=/home/hjuser/Desktop/fpga-tester-ISCAS
```

恢复时直接调用公共 `fpga_run.sh`，不要使用会附加上传段的 `run_model.sh`。

### 公共上板工具选项

服务器和工作目录也可由环境变量配置；命令行选项优先：

```bash
export FPGA_SSH_HOST=fpga
export FPGA_REMOTE_DIR=/home/hjuser/Desktop/fpga-tester-ISCAS
"$QWEN_MODEL/tools/run_model.sh" "$QWEN_NONE/prepared" \
  --fpga=5 --capture-seconds=1800 --startup-timeout=900
```

hello 只用于独立检查平台 UART／加载链路，不是 Qwen3 验收：

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/hello/build/hello.bin \
  --fpga=5 --remote-dir=/home/hjuser/Desktop/fpga-tester-ISCAS --capture-seconds=60
```

`--remote-dir` 接受相对路径或用户指定的绝对路径；不要写 `~/`。
脚本将该目录作为本次运行的写入边界，检查输出路径不会经软链接跳出该目录。
已有的 `Makefile`、`user_script`、`hw.dat` 软链接只作为平台输入使用。

- `--fpga=0` 至 `--fpga=7` 选择板卡及 `/dev/FPGAN`。
- 默认 `--baud=115200`，8N1、无流控；在加载前开启 UART 接收。
- 默认 `--startup-timeout=180`、`--capture-seconds=10`；后者从程序启动后计时。
  模型等长任务应显式增加这两个值。
- UART 内容实时写到标准输出，状态写到标准错误；可使用 `> uart.log`
  单独保存程序输出。
- `--uart-send=$'hello\n' --uart-send-delay=20` 发送预设原始文本；可重复
  `--uart-send`，用 `--uart-send-gap=5` 指定间隔。发送计划必须早于捕获截止时间。
- `--interactive` 从本机 stdin 转发原始字节，等待 FPGA 提示符后输入。
  只有已有 worker 打开 UART；主机只写入本次 run 目录的请求队列。EOF 只结束
  输入，`Ctrl-C` 仍停止本次 FPGA 会话。通常需显式设置较长的捕获时间。
  `--resume-run=run-实际编号 --interactive` 可恢复同一会话和未确认的输入请求，
  不重新上传或加载。每条请求有持久 UUID，SSH 重试不会重复发送已接受的请求。
  远端 `input/*.ack.json` 只确认操作系统接受了 UART 字节，不证明板上接收或分词
  成功；worker 崩溃后不会自动重启或重放部分输入。
- `--layout-plan=ddr-load.plan --segment=weights.bin` 使用多段 DDR 装载，可重复
  `--segment`。计划中的文件必须是本次上传文件的 basename，boot 必须叫
  `image.bin`，文件大小和 SHA256 必须完全匹配。每个声明的段都必须有完整 DDR
  读回；任意段缺失都判失败。
- 每次独立上传到服务器工作目录内的 `fpga-runs/run-*/`，检查上传 SHA256，
  不复用旧的补齐镜像。结束时检查补齐文件与 DDR 读回的一致性。
- 已有 UVHS／minicom 会话或板卡占用时会报错，不结束其他任务。
  同一工作目录同时只允许一个该脚本实例运行。
- SSH 连接和上传失败默认重试5次，可用 `--retries=10 --retry-delay=3` 调整。
  远端有时限的工作进程独立保存日志，SSH断连后继续采集；重连按UART字节位置
  接续输出，不重复复位或加载程序。达到重试次数仍无法连接时返回非零，远端工作
  进程在原定启动/捕获时限内结束，日志保留在本次目录。可用相同image、板号和
  `--resume-run=run-实际编号` 重新连接已有任务，不再次上传或执行。
- `Ctrl-C` 写入本次运行的停止请求；正常结束后自动退出 UVHS、释放串口。
  若网络完全不可达，停止请求无法送达时会明确提示，远端时限仍然有效。
- 本地日志存于本目录 `build/fpga-runs/run-*/`，远端日志留在本次上传目录。
  文件包括 `uart.raw.log`、`uvhs.log`、`worker.log` 和 `result.json`。
- 加载错误、读回不同、无 UART 输出、`verify ...: FAIL` 或 trap 返回非零。
  返回 0 说明加载和捕获流程成功，不替任意程序验证计算结果。
  通用入口不因任意 `PASS` 文本提前退出；上述模型入口已配置
  `--completion-marker='[nr] RA returned:'`，看到完成标记后会自动收尾。

脚本只负责上传和运行，不编译传入文件。需传入适配 NR 平台的 flat binary，
镜像是否已按 64 字节补齐均可；服务器包装脚本负责所需 padding。
Bash 入口与 `tools/` 下两个 Python 实现需一起保留；依赖 Bash、OpenSSH、
本机 Python 3.11+ 标准库（TOML 解析）、服务器 Python 3.8+，不依赖 pyserial 或 expect。

## 公共源码与验证

UART 和启动／运行时的来源及适配范围见
[`common/README.md`](common/README.md)。hello 的地址布局和输出说明见
[`hello/README.md`](hello/README.md)。

从仓库根目录运行离线脚本测试（模拟串口和加载进程，不操作 FPGA）：

```bash
python3 -B -m unittest discover \
  -s examples/FPGA-BOSCAME/tools -p 'test_fpga*.py' -v
```

## Qwen3-0.6B 算子

[qwen3-0.6b](qwen3-0.6b/README.md) 提供逐算子的 linalg → Buddy lowering →
RISC-V 对象 → 裸机二进制流程，覆盖 FP32 模型公式及独立的 NR W8A8 部署路径。
算子通过 [common/nr](common/nr/README.md) 的 NH→RA 运行时执行；共享 UART 和
上传脚本继续位于本目录。构建不需要本地参考仓库。

较长的算子集合可传入 `--completion-marker='[nr] RA returned:'`：脚本在接收到
该标记后保留短暂收尾时间并退出；若捕获时间结束仍缺少标记，则判为失败。
标记只控制结束时机，程序打印 FAIL/trap 或 DDR 读回不一致仍会失败。

## Triton 前端

Qwen算子的 [Triton 版本](qwen3-0.6b/triton/) 使用独立的 `triton-riscv`
编译器将真实 `@triton.jit` 内核导出为TTIR、linalg，再进入本仓库Buddy lowering。
两种前端复用同一套C launch、NR运行时和数值参考。
