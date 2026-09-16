# FPGA-BOSCAME 裸机样例与公共资源

面向 NR／RAV0.5 FPGA 平台。当前 `hello` 由 NH 单核运行，适合先验证
编译、DDR 加载和 UART 输出。其他样例可复用下列驱动、运行时及上板工具。

```text
FPGA-BOSCAME/
├── common/
│   ├── toolchain.mk       # LLVM 工具选择，可通过 make 参数覆盖
│   ├── uart/              # NR UART 驱动
│   └── runtime/           # 单核 CRT、CSR 定义及 C 运行时
├── fpga_run.sh            # 公共上板入口
├── tools/                 # SSH 上传、远端执行、离线测试
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

## 上传并运行

从仓库根目录执行：

```bash
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/hello/build/hello.bin --fpga=5
```

脚本使用 SSH 别名 `fpga`，默认进入服务器登录目录下的
`Desktop/fpga-tester-ISCAS`，在该目录内调用 `make uv_run5`。
该默认值与此前测试的服务器工作目录相对应，不含用户名或固定家目录。
SSH 登录需使用免交互密钥认证，服务器需已有 UVHS 环境和配套平台 Makefile。

可配置服务器和工作目录：

```bash
export FPGA_SSH_HOST=fpga
export FPGA_REMOTE_DIR=Desktop/fpga-tester-ISCAS
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/hello/build/hello.bin --fpga=5

# 命令行优先于环境变量；相对路径以服务器 SSH 登录目录为起点。
examples/FPGA-BOSCAME/fpga_run.sh \
  examples/FPGA-BOSCAME/hello/build/hello.bin \
  --fpga=5 --remote-dir=Desktop/fpga-tester-ISCAS --capture-seconds=60
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
- 每次独立上传到服务器工作目录内的 `fpga-runs/run-*/`，检查上传 SHA256，
  不复用旧的补齐镜像。结束时检查补齐文件与 DDR 读回的一致性。
- 已有 UVHS／minicom 会话或板卡占用时会报错，不结束其他任务。
  同一工作目录同时只允许一个该脚本实例运行。
- `Ctrl-C` 请求清理本次运行；正常结束后自动退出 UVHS、释放串口。
- 本地日志存于本目录 `build/fpga-runs/run-*/`，远端日志留在本次上传目录。
  文件包括 `uart.raw.log`、`uvhs.log` 和 `result.json`。
- 加载错误、读回不同、无 UART 输出、`verify ...: FAIL` 或 trap 返回非零。
  返回 0 说明加载和捕获流程成功，不替任意程序验证计算结果。
  `PASS` 不会自动提前结束捕获窗口。

脚本只负责上传和运行，不编译传入文件。需传入适配 NR 平台的 flat binary，
镜像是否已按 64 字节补齐均可；服务器包装脚本负责所需 padding。
Bash 入口与 `tools/` 下两个 Python 实现需一起保留；依赖 Bash、OpenSSH、
Python 3 标准库，不依赖 pyserial 或 expect。

## 公共源码与验证

UART 和启动／运行时的来源及适配范围见
[`common/README.md`](common/README.md)。hello 的地址布局和输出说明见
[`hello/README.md`](hello/README.md)。

从仓库根目录运行离线脚本测试（模拟串口和加载进程，不操作 FPGA）：

```bash
python3 -B -m unittest discover \
  -s examples/FPGA-BOSCAME/tools -p 'test_fpga*.py' -v
```
