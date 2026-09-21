# K3 端到端流程(推理only版)

> Issue #888 · Code Review 2026-09-22
> **核心分工:x86 只做一次"模型导入"(唯一需要 torch 的环节);K3 只做"编译 + 推理 + 测速",不需要任何 python 包。**

---

## 0. 为什么 K3 不需要 python 环境

| 环节 | 需要什么 | 在哪做 |
|---|---|---|
| PyTorch 模型 → MLIR 图 + 权重 | torch / transformers | **x86 一次** |
| MLIR → .o → .so | mlir-opt / buddy-opt / llc / clang++ | K3(纯 shell) |
| RHAL manifest | *纯文本*,x86 预生成;应急时系统 python3 也能生成(纯标准库) | x86 预生成 |
| 打包 .rax | `rax-pack`(**原生 C++ 二进制**,K3 buddy 构建自带) | K3(纯 shell) |
| 推理 / 测速 | buddy-cli / buddy-server + curl | K3 |

**结论:K3 上一条 `build_seq.sh` 就是一个 shell 脚本,不装任何 python 包。**

---

## 1. 目录结构

```text
~/buddy-k3/buddy-mlir/     # 源码 + 构建(已就绪)
└── examples/BuddyBgeM3/              # ★ 本任务目录(在 repo 内)
    ├── k3/                # ★ K3 侧脚本(纯 shell)
    │   ├── env.sh
    │   ├── build_runtime.sh   # 一次性:编译运行时/插件 → .so
    │   ├── build_seq.sh       # 每档:MLIR → .o → .so → .rax(优化实验的旋钮在这)
    │   ├── bench.sh           # 口径 A/B 测速
    │   ├── profile.sh         # RVV 指令普查 + VLEN
    │   └── cos.py             # 余弦(纯标准库)
    ├── src/               # ← 从 x86 传来
    │   ├── arg0.data          # 权重(共享,一份)
    │   ├── tokenizer.json     # 共享
    │   └── seq{128,256,512}/{forward.mlir,subgraph0.mlir,generated/bge_m3.mlir,seq*.json}
    ├── out/               # 构建产物 seq<L>-<tag>/
    ├── results/           # 测速原始数据 seq<L>-<tag>/
    ├── profile/           # RVV 指令统计
    └── 报告-K3.md         # ★ 正式报告(提交物)
```

---

## 2. 第一步:x86 导出(一次性)

> **权重已就绪,无需重新下载**:`$LOCAL_BGE_M3=/home/user/buddy-models/bge-m3`
> (`pytorch_model.bin` 2.2GB,2026-09-05 下载)。

```bash
# ===== 在 WSL / x86 上 =====
cd /home/user/buddy-mlir
bash examples/BuddyBgeM3/x86/export_artifacts.sh                 # 默认 128 256 512
# 或只做一档: bash examples/BuddyBgeM3/x86/export_artifacts.sh 128
scp -r examples/BuddyBgeM3/dist user@k3-003:~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/src
```

> 该脚本用"等 import 完成就停"的方式跳过 x86 上耗时的 `llc`,每档约 3–8 分钟。
> 产物里 **`arg0.data` 只存一份**(2.27GB,三档共用);`forward.mlir`/`subgraph0.mlir` 随 seq 形状不同,每档一份。

**📝 记录**:模型 repo、文件清单与大小、x86 上 buddy commit。

---

## 3. 第二步:K3 编译运行时(一次性)

```bash
# ===== 在 K3 上 =====
source ~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/k3/env.sh
bash ~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/k3/build_runtime.sh
```

产出 `out/runtime/`:`libbuddy_models_bge_m3.a`、`bge_m3_runner.so`、`bge_m3_embedding.so`。
三个 seq 共用,只需跑一次。

**📝 记录**:K3 上 `llc --version`、clang 版本、cmake/ninja 版本(进报告"环境"节)。

---

## 4. 第三步:K3 编译 rax(每档一次)

```bash
source ~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/k3/env.sh
bash $APP/k3/build_seq.sh 128                # → out/seq128-baseline/
bash $APP/k3/build_seq.sh 256
bash $APP/k3/build_seq.sh 512
```

**耗时警告**:subgraph0 的 `llc -O3` 是大头,单档可能 10–40 分钟(取决于内存/核数)。
建议后台跑 + `tail -f`:`nohup bash build_seq.sh 128 > out/seq128-build.log 2>&1 &`

**📝 记录**:每档编译时间、产物大小(`arg0.data`≈2.27GB、`.rax`≈2.3GB)。

---

## 5. 第四步:正确性 + 基线(核心交付物)

```bash
# 先跑一次 128 拿基线 embedding(兼冒烟)
bash $APP/k3/bench.sh 128 baseline 20 10
cp $APP/results/seq128-baseline/emb.txt $APP/results/seq128-baseline/emb_baseline.txt

# 256 / 512
bash $APP/k3/bench.sh 256 baseline 20 10
bash $APP/k3/bench.sh 512 baseline 20 10
```

`bench.sh` 输出:
- `latency_b.txt` 稳态逐次(ms→ 汇总 median/stdev)
- `latency_a.txt` 冷启动逐次(wall_s + maxRSS_KB)
- `emb.txt` 本次 embedding
- `summary.txt` 汇总

**正确性门**:基线本身对 HF 已验证(x86 阶段 cos=1.000000);之后每次优化用
`python3 k3/cos.py 基线/emb_baseline.txt 新实验/emb.txt`,**>0.999 才允许记录性能**。

**📝 记录**(填进 `报告-K3.md` 基线表):

| seq | 精度 | 稳态 ms | seq/s | token/s | 冷启动 ms | 峰值内存 MB | 核/线程 | 备注 |
|---|---|---|---|---|---|---|---|---|

---

## 6. 第五步:RVV 普查 → 优化实验

```bash
bash $APP/k3/profile.sh 128 baseline
```

**判读**:
- `vfmacc / vle32.v` **大量** → 瓶颈在带宽/线程 → 先做:
  `OMP_NUM_THREADS=1 vs 16`、`taskset -c 0-15`、`profile.sh` 量出的 VLEN 加进 `LLC_ATTRS`
- **几乎没有** → 做 scalable 向量化实验 ⬇️

### 优化实验(改一个 shell 变量,重跑)

编辑 `k3/build_seq.sh` 里的 `SUB_PASSES`,然后:

```bash
source ~/buddy-k3/buddy-mlir/examples/BuddyBgeM3/k3/env.sh
bash $APP/k3/build_seq.sh 128 exp1_scalable

# ① 先过余弦门(单次推理,几秒~几十秒)
$BUDDYBIN/buddy-cli --model $OUT/seq128-exp1_scalable/bge_m3.rax \
  --prompt "hello world" --no-stats > /tmp/emb_exp1.txt
python3 $APP/k3/cos.py \
  $APP/results/seq128-baseline/emb_baseline.txt /tmp/emb_exp1.txt   # 必须 >0.999

# ② 过了门再测速 + 数指令
bash $APP/k3/bench.sh   128 exp1_scalable
bash $APP/k3/profile.sh 128 exp1_scalable
```

**实验阶梯**(每次只改一件事,过了 cos 门再进下一级):

| # | 改动 | 目的 |
|---|---|---|
| C | 只加 `-eliminate-memref-copy -assume-tight-memref-layout -staticize-memref-layout` | 验证"布局是数值错误根因" |
| A | 换 `-matmul-vectorization="vector-type=scalable vector-size=4"` + **删** `-convert-vector-to-scf` | RVV 可伸缩向量化 |
| B | A + `-batchmatmul-transpose-b-vectorization="vector-type=scalable vector-size=4" -batchmatmul-optimize` | 覆盖 48 个 batch_matmul |
| D | `vector-size=4 → 8` | 展开度 |
| E | `LLC_ATTRS` 加 VLEN 提示(如 `-riscv-v-vector-bits-min=256`) | 后端向量宽度 |

**为什么不要再用 x86 那三次的 blis**:`matmul-vectorization-blis` 和
`matmul-parallel-vectorization-optimize` **都没有 `vector-type` 选项**(不支持 scalable),
是 x86/AArch64 定长向量方向;RVV 要用 `vector<[4]xf32>` 形态。
另外 `VectorToSCF` **不支持 scalable**,所以用 scalable 时必须删 `-convert-vector-to-scf`。

**📝 记录**(每次实验):

| 实验 | diff | cos | RVV 指令数(前→后) | 稳态 ms(前→后) | 提速 | 结论 |
|---|---|---|---|---|---|---|

---

## 7. 记录清单 → issue 交付物

| 记录项 | 放哪 | 对应 |
|---|---|---|
| 环境/配置 | `报告-K3.md` §1 | 可复现 |
| 128/256/512 基线表 | §2 | Deliverable 1、3 |
| 参考对比(x86 已有 HF 数据,注明非同机;或用 K3 上的 torch 补测) | §3 | Deliverable 2 |
| RVV 指令普查 + VLEN + 瓶颈排序 | `profile/` + §4 | Deliverable 3、5 |
| 每次优化 diff/cos/Before-After | `results/exp*` + §5 | Deliverable 4、6 |
| 三方对比表 | §6 | Deliverable 6 |
| IME 段落(接入 or 量化证据) | §7 | Deliverable 4 |
| 剩余差距与后续 | §8 | 量化证据要求 |

---

## 8. 排错

| 现象 | 原因 / 处理 |
|---|---|
| `build_seq.sh` 报缺少 `$SRC/...` | 没传 `dist`,或目录没对上:`scp -r dist .../examples/BuddyBgeM3/src` 后结构应为 `src/seq128/...` |
| 找不到 `libomp.so` | K3 的 LLVM 需以 `-DLLVM_ENABLE_RUNTIMES=openmp` 构建;确认 `llvm/build` 里有 |
| `bge_cli` 报 `zero/invalid CLS` | 数值错误 → 先回退到 baseline,再只改一个变量重试 |
| `rax-pack` 不存在 | `ninja -C $BUILD rax-pack` |
| 编译 OOM | 同机跑其它大进程;或先编小档(128) |
| 想看 scalable 是否命中 | `bash build_seq.sh 128 t1` 后 `grep -c 'vector<\[4\]xf32'`(需先导出中间 MLIR) |
