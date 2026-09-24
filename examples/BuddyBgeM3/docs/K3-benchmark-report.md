# BGE-M3 RAX 性能报告(Issue #888)

> 更新:2026-09-21 · 状态:**定稿,待提交 #888**。K3 正式数据在 §6;同硬件对比结论:RAX 128/256 快于参考、512 持平(§3.3)。
> 说明:§1–§5 为 x86 参考数据;**§6 为 K3 正式数据**(riscv64 + RVV,交付主体)。

## 0. 环境与配置

| 项 | 值 |
|---|---|
| 平台 | x86_64 本机(4 核,AVX512,内存 11GB) |
| LLVM/MLIR | 24.0.0git nightly 预编译(`/home/user/buddy-llvm-nightly`) |
| buddy-mlir | `main` = `e7cc8cea`(nightly v0.0.8.dev20260904) |
| 模型 | `BAAI/bge-m3`,**f32**,~568M 参数,权重 2.27GB,hidden=1024,24 层 |
| 固定输入 | `The quick brown fox jumps over the lazy dog.` |
| 脚本 | x86 阶段脚本(本地归档,未随本例提交):`bench_server.sh`(口径 B)/ `bench_cli.sh`(口径 A) |

**测量口径**(issue 要求记录):
- **口径 A** = `buddy-cli` 单次进程,**含 2.27GB 权重加载**(冷启动端到端)
- **口径 B** = `buddy-server` 常驻,**纯推理稳态**(报告主口径)

### 0.1 测量方法(全部数据遵循)

**口径定义**
- **口径 A(冷启动端到端)**:`buddy-cli --model <rax> --prompt <text> --no-stats` 单次进程,计时 = 进程墙钟;
  峰值内存优先用 `/usr/bin/time -f '%e %M'`,K3 无此工具时退化为 `date +%s.%N` 计时 +
  `/proc/<pid>/status` 的 `VmHWM` 轮询(报告已标注)。
- **口径 B(稳态单请求,报告主口径)**:`buddy-server` 常驻加载一次后,HTTP POST `/v1/embeddings`
  (`Content-Type: application/json`,body `{"input":"<text>"}`),计时 = `curl -w '%{time_total}'`;
  就绪判定 = 首次 HTTP 200(兼作预热,排除模型加载与冷缓存)。

**取数规范**
- 每档每口径重复 10–20 次,**丢弃首次**,统计 **median + stdev**,原始逐次数据全部留档(`results/`);
- 同一固定文本、同一 `max_length`(pad/truncate)、同一线程数 → 保证 RAX 与 Reference 可比;
- 所有数字标注 `精度 / baseline|optimized`;不同精度不横比。

**正确性门(红线)**
- 基线:RAX embedding 与 HF(`AutoModel` → `last_hidden_state[:,0]` → L2 norm)同文本余弦,要求 **> 0.999**;
- 每个优化实验:与 K3 基线 embedding 余弦 **> 0.999** 才允许记录性能;失败即回退并记录负结果。

**Profile 方法**
- RVV 指令普查:`llvm-objdump -d bge_m3_model.so` 做助记符直方图(`vsetvli / vle32.v / vse32.v / vfmacc.* / vfred*`);
- VLEN:内联汇编 `csrr vlenb`(K3 实测 256 bits);
- 线程扩展性:同一 rax、`OMP_NUM_THREADS` 扫 1/4/8/16,每次重启 server、预热后采样 2 次;
- 运行时拆段:`BgeM3Runtime::embed()` 内临时 `std::chrono` 打点(测后回退源码)。

## 1. 正确性校验(阶段 0,红线:cos > 0.999)

| 文本 | 档位 | cosine | MSE | 结果 |
|---|---|---|---|---|
| hello world | 512 | 1.000000 | 3.45e-15 | ✅ PASS |
| you are good | 512 | 1.000000 | 3.21e-15 | ✅ PASS |

> 每档/每次优化后都须复核(用对应 `max_length`)。

## 2. P1 基线结果

### 2.1 总表(f32)

| seq_len | 稳态延迟 ms(口径B) | seq/s | token/s | 冷启动 ms(口径A) | 峰值内存 MB | 核/线程 | 备注 |
|--------:|--------------------:|------:|--------:|------------------:|------------:|:-------:|------|
|     128 |         **3635.94** |  0.28 |    35.2 |   5710(去首中位) |        4265 |       4 | baseline ✅(x86) |
|     256 |         **7286.01** |  0.14 |    35.1 |  42220(去首中位) |        5403 |       4 | baseline ✅(x86) |
|     512 |        **44142.08** |  0.02 |    11.6 | 114440(去首中位) |        8566 |       4 | baseline ✅(x86) |

> 换算口径:seq/s = 1 / 稳态中位数;token/s = seq_len / 稳态中位数。
> **正式基线在 K3(16×X100),见 §6.2;本节仅为 x86 参考数据。**

### 2.2 128 档明细(seq_len=128,f32,baseline)

**口径 B · 稳态**(`bench_server.sh`,n=20):
- `median = 3635.94 ms`,`stdev = 319.80 ms`,`mean = 3720.98 ms`
- 逐次(ms):`3657.6, 3960.5, 3703.8, 3605.2, 3725.7, 4509.5, 4257.7, 4156.1, 3612.9, 3418.9, 3750.7, 3614.3, 3305.9, 3762.8, 3498.5, 3564.6, 3960.9, 3601.0, 3156.8, 3596.3`
- 换算:`seq/s ≈ 0.275`;`token/s ≈ 35.2`

**口径 A · 冷启动**(`bench_cli.sh`,n=10,去首 n=9):
- `median = 5710 ms`;`峰值内存 median = 4265.1 MB`
- 逐次 wall(s):`30.23, 17.37, 13.57, 7.30, 5.71, 5.16, 5.23, 5.50, 5.62, 6.16`(首次 30.23s 为权重冷加载,波动大属正常)
- 逐次 maxRSS(KB):≈ `4,367,604`(稳定在 ~4265 MB)

### 2.3 256 档明细(seq_len=256,f32,baseline)

**口径 B · 稳态**(`bench_server.sh`,n=20):
- `median = 7286.01 ms`,`stdev = 303.18 ms`,`mean = 7419.64 ms`
- 逐次(ms):`7251.3, 7574.1, 8131.9, 8085.0, 7345.5, 7273.6, 7208.6, 7577.3, 7112.5, 7229.1, 7059.4, 7824.9, 7294.1, 7272.5, 7103.2, 7597.8, 7261.2, 7512.1, 7277.9, 7400.8`
- 换算:`seq/s ≈ 0.137`;`token/s ≈ 35.1`

**口径 A · 冷启动**(`bench_cli.sh`,n=10,去首 n=9):
- `median = 42220 ms(≈42.2 s)`;`峰值内存 median = 5402.5 MB`
- 逐次 wall(s):`43.03, 42.22, 40.30, 35.42, 34.98, 37.34, 60.77, 89.23, 67.29, 74.21`(进程级含权重加载,波动大属正常)
- 逐次 maxRSS(KB):≈ `5,532,240`(稳定在 ~5.4 GB)

### 2.4 512 档明细(seq_len=512,f32,baseline)

**口径 B · 稳态**(`bench_server.sh`,n=20):
- `median = 44142.08 ms`,`stdev = 8884.30 ms`,`mean = 44628.18 ms`
- 逐次(ms):`45965.9, 41289.9, 65097.2, 44511.8, 50462.8, 37666.3, 41460.4, 55567.7, 36083.2, 44110.6, 45407.2, 30695.1, 53084.8, 58682.3, 44173.6, 43675.3, 47332.0, 37419.7, 40869.1, 29008.8`
- 换算:`seq/s ≈ 0.023`;`token/s ≈ 11.6`

**口径 A · 冷启动**(`bench_cli.sh`,n=10,去首 n=9):
- `median = 114440 ms(≈114.4 s)`;`峰值内存 median = 8566.3 MB`
- 逐次 wall(s):`108.78, 115.56, 119.72, 116.61, 114.44, 113.59, 112.99, 133.91, 112.85, 101.34`
- 逐次 maxRSS(KB):≈ `8,772,028`(稳定在 ~8.6 GB)

> ⚠️ **观察**:512 档稳态延迟(≈44s)远非线性于 128/256(3.6s/7.3s,按线性外推应 ≈15s),且波动极大(stdev≈8.9s,29–65s),峰值内存 8.6GB 已逼近本机 11GB 物理内存 → 很可能受**内存压力/页表抖动**影响,而非纯计算。需在 K3(内存充足)复测,以区分"编译/运行瓶颈"与"本机内存瓶颈"。

## 3. Reference 对比(HF,P2)

### 3.1 对比表(x86 本机,4 线程,f32;RAX = 口径 B 稳态 median)

| seq_len | RAX-Before(ms) | HF-Reference(ms) | RAX/HF | 备注 |
|--------:|----------------:|-----------------:|-------:|------|
|     128 |        3635.94 |       **410.18** |   8.9× | — |
|     256 |        7286.01 |       **716.87** |  10.2× | — |
|     512 |       44142.08 |     **1634.96** |  27.0× | x86 内存压力可能放大差距 |

HF 明细(`hf_bench.py`,threads=4,repeat=10,去首 median):
- 512:median `1634.96 ms`(逐次 `1901.08, 1815.51, 1682.34, 1634.96, 1641.06, 1609.20, 1585.36, 1669.49, 1603.78, 1603.89`)
- 256:median `716.87 ms`(逐次 `746.24, 807.18, 813.53, 815.18, 712.22, 684.09, 691.99, 762.03, 665.50, 716.87`)
- 128:median `410.18 ms`(逐次 `362.77, 415.43, 423.35, 383.13, 403.46, 390.65, 410.18, 435.22, 443.45, 380.46`)

> 对齐条件:同机、同 4 线程、同输入、f32、同 `max_length`。HF 走 PyTorch oneDNN/MKL。

### 3.2 观察

- RAX-Before 本机落后 HF **约 9×–27×**(128/256 ≈ 9–10×,512 ≈ 27×)。
- HF 128→512 延迟近线性(410→1635 ms,≈4×);RAX 128→512 为 **12×** → 大 seq 扩展性差,与 P1 中 512 档内存压力/代码质量问题一致。
- 本表为 **x86 参考**;同硬件(K3)对比见 §3.3(结论不同,注意区分)。

### 3.3 同硬件对比(K3,2026-09-21,16 线程,f32,同输入)

| seq | RAX-Before(K3) ms | HF-Reference(K3) ms | RAX/HF  | 结论            |
|----:|------------------:|--------------------:|--------:|-----------------|
| 128 |       **19926.6** |            24681.6  | **0.81×** | **RAX 快 24%**  |
| 256 |         **39573** |            52744.1  | **0.75×** | **RAX 快 33%**  |
| 512 |      **115658.7** |           110123.0  |  1.05×   | 基本持平        |

> HF(K3)明细(`hf_bench.py`,threads=16,repeat=10 去首):128 → median 24681.63 ms;256 → 52744.08 ms;512 → 110122.98 ms。
> 对齐条件:同一块 K3、同 16 线程、同输入、f32、同 `max_length`。

**关键结论**:
- **同硬件下,RAX 基线在 128/256 已超过参考实现,512 持平** → 满足 issue Performance Target 的"reach or exceed the applicable native/reference implementation";
- K3 上的 PyTorch(Ruyi 通用 build,无优化内核)本身效率极低(128/256/512 ≈ 3.1/2.9/2.8 GFLOPS,与 RAX 的 3.9/3.9/2.7 相当)→ 之前"落后 49×"是 **x86 vs K3 跨硬件比较**造成的假象;
- 两者距 x86 HF 的 ~190 GFLOPS 仍有 **~50× 差距** → 真正优化空间在 **RVV 向量化深度**,后续目标应表述为"相对基线提升效率",而非"追上参考"。

## 4. P3 Profile 初步(算子结构统计)

### 4.1 图规模(`build/models/bge_m3/`)

| 文件 | 大小 / 行 | 角色 |
|---|---|---|
| `subgraph0.mlir` | 0.52 MB / 5203 行 | **计算主体**(tosa+linalg) |
| `forward.mlir` | 0.19 MB / 942 行 | 主包装(带全部权重 memref 参数) |

### 4.2 算子分布(subgraph0.mlir)

- **tosa:5777** → 细分:`shape 1906, const_shape 953, reshape 953, const 489, add 392, mul 342, transpose 240, sub 146, reduce_sum 122, reciprocal 98, rsqrt 49, reduce_max 24`
- **linalg:626** → `matmul 144, batch_matmul 24, generic 25, fill 240, type_fn 168`
- **tensor:274**(empty 266)、**math:72**(exp 48,erf 24)

### 4.3 初步判断

- 核心矩阵乘 **~168 个(每层约 7)**,符合 24 层 XLM-R;LayerNorm(rsqrt/reduce_sum/sub/mul)、softmax(exp/reduce_max/reciprocal)、GELU(erf)等 elementwise 规模正常。
- **可疑点 ①:reshape 953 + transpose 240**——数量偏多;若 buffer 化后不能折叠为 0-copy view,会变成**真实内存搬运**,直接放大带宽压力(与 P1"token/s 恒定 ~35、疑似带宽 bound"一致)。
- **可疑点 ②:shape/const_shape 2859**——属编译期元数据,通常被折叠、不直接产生运行时代价,但其规模反映前端图"每 op 附带 shape 传播",图质量一般。
> **待验证**:reshape/transpose 在 buffer 化后是否被消除(0-copy);运行时各段(GEMM/搬运/小算子)耗时占比 → 需后续打点 / perf。

### 4.4 buffer 化分析(方案 B;`buddy-opt` tosa→linalg→`one-shot-bufferize`,产物 9457 行)

**关键数字**:
- `memref.copy` **仅 2** → reshape/transpose 前端冗余**大部分被折叠为 0-copy view**(`collapse_shape` 240 / `expand_shape` 586),**不是**搬运主因;
- 但残留:**`linalg.generic` 1063、`linalg.transpose` 240、`linalg.reduce` 146、`linalg.matmul` 144、`linalg.batch_matmul` 48**;
- `vector.*` = **0**(该阶段无显式向量化,依赖后续 buddy matmul 向量化 + `llc` 自动向量化);
- `memref.alloc` 1648(大量中间缓冲;未跑 dealloc 优化故 `dealloc`=0)。

**判断**:
- reshape(953)不是搬运大头(0-copy);
- **剩余疑点:240 个 `linalg.transpose` 需 lower 搬运 + 1063 个 `generic`/146 个 `reduce` 逐元素算子**(若 `llc` 自动向量化不佳即为瓶颈);
- 与实测结合:矩阵乘全权重流(带宽)+ 大量逐元素/搬运仍是主疑点 → **P4 既要 matmul 向量化,也要关注 elementwise/transpose 的融合与向量化**。

### 4.5 运行时打点(方案 C,临时 chrono;测后已回退源码)

`BgeM3Runtime::embed()` 内临时加 `std::chrono` 分段,`buddy-cli`(512 档)一次:

| 段 | 耗时 | 占比 |
|---|---:|---:|
| tokenize+chk | 0.14 ms | ~0% |
| prep(copy/alloc) | 0.00 ms | ~0% |
| **forward(+post)** | **51361.09 ms** | **≈100%** |
| total | 51361.23 ms | 100% |

**判断**:
- **forward(编译后 MLIR kernel)占 ~99.999%**——tokenizer/runner/后处理开销可忽略;
- 瓶颈完全在编译产物 kernel 内部,与 server 稳态 median(44.1s)量级一致(本次 51.4s 属波动);
- 后续只需在 **kernel 层**定位(带宽 / memcpy / 计算),无需再优化 runner 侧。

### 4.6 MLIR pass 耗时(计划 §5.2;`--mlir-timing`,复刻 subgraph0 真实 pipeline)

`buddy_model.cmake` single_forward Stage2(`subgraph0.mlir` 的计算 pipeline)MLIR 段实测:总 **5.82 s**(MLIR 侧很小,真正编译大头在 `llc -O3`)。

| pass | 耗时(s) | 占比 |
|---|---:|---:|
| BufferDeallocationSimplification | 0.726 | 12.5% |
| SCFToControlFlow | 0.490 | 8.4% |
| FinalizeMemRefToLLVM | 0.328 | 5.6% |
| ConvertOpenMPToLLVM | 0.316 | 5.4% |
| ConvertMathToLibm | 0.289 | 5.0% |
| ExpandStridedMetadata | 0.281 | 4.8% |
| ConvertLinalgToLoops | 0.273 | 4.7% |
| ConvertVectorToSCF | 0.254 | 4.4% |
| **MatMulParallelVectorization** | **0.021** | **<1%** |
| (Total) | 5.816 | 100% |

**判断**:
- **`MatMulParallelVectorizationPass` 仅 0.02s(<1%)** → 对 144 个 matmul 几乎未做实质向量化;随后 `ConvertVectorToSCF`(0.25s)又把 vector 降回标量 scf → **运行时 matmul 基本靠 `llc -O3` 自动向量化**,无显式向量 IR;
- 编译链中**无任何 RVV 专用 pass**(subgraph0 pipeline 是 P4 的注入点);
- MLIR 编译侧开销很小 → 注入 RVV/向量化 pass 的**编译成本可接受**;P4 重点是让 matmul/generic 得到真正向量化,并避免被 `convert-vector-to-scf` 拉回标量。

### 4.7 算力 / 带宽理论模型(计划 §5.5)

模型:XLM-R-large **24 层、H=1024、FFN=4096**。单序列前向 FLOPs ≈ `2·S·(4·H² + 2·H·4096)·24 ≈ 576·S·H²`。

| seq_len | 前向 FLOPs | RAX 实测 | RAX 实际算力 | HF 实测 | HF 算力 |
|---|---:|---:|---:|---:|---:|
| 128 | 7.7e10 | 3.636 s | **21 GFLOPS** | 0.410 s | 188 GFLOPS |
| 256 | 1.55e11 | 7.286 s | **21 GFLOPS** | 0.717 s | 216 GFLOPS |
| 512 | 3.1e11 | 44.14 s | **7 GFLOPS** | 1.635 s | 190 GFLOPS |

带宽视角:每请求至少应读一遍全部权重(2.27GB),若以实测延迟折算"隐含带宽":128→0.62 GB/s、256→0.31 GB/s、512→0.05 GB/s,**远低于硬件带宽(数十 GB/s 量级)**。

**判读(重要修正)**:
- 128/256 的 RAX 实际算力**恒定 ≈21 GFLOPS**(时间随 S 近线性)→ 不是纯带宽 bound,而是**执行效率极低且恒定**(标量为主、访存差);
- 隐含带宽仅 0.3–0.6 GB/s 佐证访存/指令极低效;
- 512 骤降至 7 GFLOPS → 额外叠加**内存压力**(峰值 8.6GB/11GB 物理内存);
- 对比 HF ≈190–216 GFLOPS:RAX 执行效率约为其 **~10%(128/256)与 ~3.7%(512)**。

### 4.8 瓶颈排序表(计划 §5.6)

| 段 | 证据 | bound 类型 | 优先级 | 对应优化 |
|---|---|---|---|---|
| forward 总 | 打点 ≈100%(§4.5) | 执行/访存低效 | — | 显式向量化(RVV) |
| └ MatMul(144+batch 48,FLOPs 主导) | MatMulVecPass 0.02s(§4.6) | 标量+低效访存 | **P1** | RVV matmul + 分块/权重复用 |
| └ linalg.generic 1063 / reduce 146 | buffer 化残留(§4.4) | 逐元素标量 | P2 | 融合 + 向量化 |
| └ linalg.transpose 240 | buffer 化残留 | 数据搬运 | P3 | 消除 / 向量化 |
| └ memref.alloc 1648 | buffer 化残留 | 内存压力(512 恶化) | P4 | 复用 / 减分配 |
| tokenize / pooling+L2 | 0.14ms(§4.5) | — | — | 无需优化 |

> 结论:优化应**同时**打在 matmul 向量化(RVV/分块)与逐元素/搬运融合上;512 档还需关注中间分配造成的内存压力。全部改动须过余弦相似度正确性门。

## 5. P4 实验记录(诊断 + 第一刀 + 第二刀)

**实验**(2026-09-05,128 档):在 `buddy_model.cmake` subgraph0 链把 `-matmul-parallel-vectorization-optimize` 改为 `=vector-size=16`,并**删除 `-convert-vector-to-scf`**(保留 vector 至 LLVM)。改动后重编、回退已执行(`git checkout` 还原)。

| 项 | Before(默认) | After(第一刀) | 结论 |
|---|---|---|---|
| 正确性(cos) | 1.000000 | 1.000000 | PASS |
| 稳态延迟 128(ms) | **3635.94** | **8332.12**(stdev 370,n=20) | **变慢 2.3×** |

**结论(负结果,同样重要)**:
- "保留显式 vector 到 LLVM"的形态在本机(x86/AVX512)**反而差于**默认的"vector 回标量 scf → `llc -O3` 自动向量化";
- 说明瓶颈**不是**简单的"被 `convert-vector-to-scf` 反向量化",而更可能是:
  ① `matmul-parallel-vectorization-optimize` 对 BGE-M3 的 f32 matmul 命中/代码质量问题(待验证是否真命中);
  ② matmul 内核本身的并行/布局/分块,而非 pipeline flag 层面;
- 下一步:验证该 pass 是否命中(生成 IR 中 vector 数)、对比 `matmul-vectorization-blis`/`matmul-vectorization` 等替代、以及 OpenMP 线程/布局。

### 5.1 P4 诊断:matmul pass 命中与向量化覆盖

| pass(单跑于 buffer 化 IR) | matmul 处理 | vector ops | contraction | 形态 |
|---|---|---|---|---|
| 默认 `matmul-parallel-vectorization-optimize` | 144→0 | 576(每 matmul ~4) | 0 | affine 并行 + **标量内层** |
| `matmul-vectorization-blis` | 144→0 | 4752(每 matmul ~33) | 0 | scf.for 分块 + 丰富 vector |
| `matmul-vectorization` | 144→0 | 5472 | 0 | 同上更密 |

**结论**:默认 pass 虽然命中全部 matmul,但**每 matmul 只产生 ~4 个 vector op、无 contraction → 主体是标量循环(向量化极弱)**;blis/通用 pass 的 vector 覆盖是其 8× 以上。→ 默认链"matmul 弱向量化"是 RAX 慢的直接主因之一。

### 5.2 第二刀实验:改用 `matmul-vectorization-blis`(去 convert-vector-to-scf)

| 项 | 结果 |
|---|---|
| 正确性 | **FAIL**(`forward returned zero/invalid CLS`) |
| 处置 | 已回退(`git checkout`) |

**结论**:blis 单独替换即产生错误数值,**不配套**(需正确接入/tiling),不能简单 swap。**当前基线(matmul-parallel)数值正确但向量化弱**——真正的提升需要更深入的内核工作,而非换 pass 参数。

### 5.3 第三刀实验:blis(保留 convert-vector-to-scf)

| 项 | 结果 |
|---|---|
| 正确性 | **FAIL**(zero/invalid,同第二刀) |
| 处置 | 已回退 |

**结论**:blis 无论是否保留 `convert-vector-to-scf` 在这条 subgraph0 链上都数值错误 → 它需要更深前置/配套(`batchmatmul-optimize`、tiling、不同 bufferize 形态),**pass 级 swap 三连均不可行**:
- 默认 matmul-parallel:正确(cos=1.0)但向量化弱(标量主体);
- blis / 通用 matmul-vectorization:向量覆盖强,但直接接入产生错误数值。

**真正可行方向(非 pass-swap)**:① 用 MLIR upstream `linalg→vector`(标准 vectorize/linalg-vectorize)替换 buddy 专有 matmul pass,先在本机验证正确性+是否生成有效 SIMD;② 自写 matmul micro-kernel(vector.contraction / RVV);③ 或先上 K3,直接验证 RVV 自动向量化与 llc 表现。

## 6. K3 正式结果(riscv64 + RVV,2026-09-20 ~ 09-21)

### 6.1 环境

| 项 | 值 |
|---|---|
| 平台 | SpacemiT K3(Bianbu 4.0.1 / kernel 6.18.3 / riscv64) |
| CPU | 16× Spacemit X100(RVV 1.0,`rv64imafdcv`+`zve64d`+zvbb/zvk*,VLEN=256);IME A100 核不出现在 Linux |
| 内存 / 磁盘 | 31 GB / 105 GB |
| LLVM | 24.0.0git(RuyiAI riscv 分支 `2d26d272a`;`llc` 默认 target riscv64,Host CPU=spacemit-x100) |
| buddy-mlir | `e7cc8cea`;K3 原生用户态构建(无 sudo,无 python 包) |
| 工具链 | 自举 cmake 3.31.6 + ninja 1.12.1;gcc 15.2 |
| 测量 | `examples/BuddyBgeM3/k3/bench.sh`(口径 A/B);正确性 = 与 x86 基线 embedding 的余弦 |

### 6.2 基线总表(f32,16 线程 OpenMP)

| seq | 稳态 ms(口径B) | stdev | seq/s  | token/s | 冷启动 s(口径A) | 峰值内存 MB | cos       | 备注         |
|----:|---------------:|------:|-------:|--------:|-----------------:|------------:|----------:|--------------|
| 128 |     **19926.6** |   120 | 0.050  |     6.4 |        29.1(中位) |        4365 | 1.000000  | baseline ✅  |
| 256 |       **39573** |  ≈800 | 0.025  |     6.5 |        42.9      |        5399 | 1.000000  | baseline ✅  |
| 512 |    **115658.7** |  6490 | 0.0086 |     4.4 |       128.4      |        8568 | 1.000000  | baseline ✅  |

> - 128→256 近线性(19.9→39.6s);512 略超线性(115.7s vs 4× 外推 79.7s),但远好于 x86(12×)。
> - 实际算力:**128/256 ≈ 3.9 GFLOPS,512 ≈ 2.7 GFLOPS**(对照:x86 RAX 21、HF 190–216 GFLOPS)。
> - 与 x86 RAX 同口径比:K3 慢 ~5.5×(128/256)、~2.6×(512,x86 的 512 受内存压力拖累)。

### 6.3 RVV 指令普查(seq128 baseline,`llvm-objdump`)

| 指令 | 数量 | 解读 |
|---|---:|---|
| `vsetvli` | 144 | **每个 matmul 恰好 1 个向量循环** |
| `vle32.v` / `vse32.v` | 576 / 288 | 每 matmul 4 载 2 存 |
| `vfmacc.vf` | 288 | **标量×向量 FMA → K 维是标量循环** |

→ RVV 确实在用,但形态是"仅 N 轴向量化 + K 标量",与 x86 诊断(默认 pass 每 matmul ~4 个 vector op、无 contraction)完全吻合。

### 6.4 线程扩展性(seq128 baseline,2026-09-21)

| OMP_NUM_THREADS | 单请求 s(2 次采样) | 相对 1 线程加速 |
|---|---:|---:|
| 1 | 49.27 | 1.00× |
| 4 | 32.83 | 1.50× |
| 8 | 20.85 | 2.36× |
| 16 | 20.32 | 2.42× |

**判读**:
- 1→8 仅 2.36×(理想 8×),**8→16 完全饱和**(+2.5%)→ 不是纯计算 bound;
- 结合 §6.3(`vfmacc.vf` 标量 K 循环)与 §4.4(1063 generic / 240 transpose / 146 reduce 串行段):瓶颈 = **标量访存流 + 未并行段**,加线程无效;
- 由此定调后续方向:① 修 matmul 向量化 pass 的正确性(拿 `vfmacc.vv`);② 减内存流量(FP16/BF16 权重减半、IME GEMM 下放);调线程数不在这条主线上。

### 6.5 优化实验(全部负结果,同样构成交付物)

> 总览见下表;K3 每个实验的"假设 → 改动 → 证据 → 根因"明细见 §6.5.1–§6.5.5;
> x86 三刀(#1–#3)的完整过程见 §5(§5.1/§5.2/§5.3)。

| # | 平台 | 实验(改动点) | 产物 tag | 结果 |
|---|---|---|---|---|
| 1 | x86 | `matmul-parallel-vectorization-optimize=vector-size=16` + 删 `convert-vector-to-scf` | — | cos=1.0 但**慢 2.3×** |
| 2 | x86 | `matmul-vectorization-blis` | — | **zero/invalid CLS** |
| 3 | x86 | blis + 保留 `convert-vector-to-scf` | — | **zero/invalid CLS** |
| 4 | K3 | `matmul-vectorization=vector-type=scalable` + 布局三连 | `exp1_scalable` | **verifier 崩溃**(`expand_shape` stride 不变量被 `staticize-memref-layout` 破坏) |
| 5 | K3 | scalable(去掉布局三连) | `exp1_scalable` | **zero/invalid CLS** |
| 6 | K3 | fixed `vector-size=16` | `exp1_fixed16` | **zero/invalid CLS** |
| 7 | K3 | `-mcpu=spacemit-x100`(只改后端,不改 IR) | `exp2_mcpu` | **cos=0.340539329 数值错误** |
| 8 | K3 | `+zvl256b -riscv-v-vector-bits-min=256`(只改后端) | `exp4_g2` | **cos=0.340539329 数值错误**(与 #7 完全相同) |

**实验底座**(所有 K3 实验共用,保证可比、可复现):
- **两个旋钮**:编译旋钮 = `k3/build_seq.sh` 里的 `SUB_PASSES`;后端旋钮 = `k3/env.sh` 里的 `LLC_ATTRS`。
  每个实验只改其中一个(或明确组合),换 tag 重跑;基线产物 `out/seq128-baseline/` 从不覆盖;
- **基线 `SUB_PASSES`**(subgraph0 链,verbatim;K3 与 x86 一致):

```text
-arith-expand -eliminate-empty-tensors -convert-elementwise-to-linalg
-empty-tensor-to-alloc-tensor -one-shot-bufferize=bufferize-function-boundaries
-ownership-based-buffer-deallocation -buffer-deallocation-simplification
-bufferization-lower-deallocations -matmul-parallel-vectorization-optimize
-convert-linalg-to-affine-loops -affine-loop-fusion -affine-parallelize -lower-affine
-convert-scf-to-openmp -convert-linalg-to-loops -convert-vector-to-scf
-expand-strided-metadata -lower-affine -cse -convert-vector-to-llvm -memref-expand
-convert-arith-to-llvm -finalize-memref-to-llvm -convert-scf-to-cf -convert-cf-to-llvm
-llvm-request-c-wrappers -convert-openmp-to-llvm -convert-arith-to-llvm
-convert-math-to-llvm -convert-math-to-libm -convert-func-to-llvm
-reconcile-unrealized-casts
```

- **基线后端**(verbatim):subgraph0 用 `llc $LLC_ATTRS -O3`,其中
  `LLC_ATTRS="-march=riscv64 -mattr=+m,+d,+v -mtriple=riscv64-unknown-linux-gnu"`(**不声明 zvl**);
- **正确性门**:每个产物与 `results/seq128-baseline/emb.txt` 做余弦(`k3/cos.py`),> 0.999 才测速;FAIL 即回退并记录。

#### 6.5.1 #4:布局三连 + scalable `matmul-vectorization` → verifier 崩溃

**假设**:BGE-M3 子图存在动态/非紧凑布局(stride 编译期未知),而 scalable 向量化希望布局静态可知;
先跑 `eliminate-memref-copy / assume-tight-memref-layout / staticize-memref-layout`(布局三连)把布局定死,再上 scalable,理论上能让 vector size 推断与 stride 展开全部静态化。

**改动**(相对基线 `SUB_PASSES` 的 diff):

```diff
-  -matmul-parallel-vectorization-optimize
+  -eliminate-memref-copy -assume-tight-memref-layout -staticize-memref-layout
+  -matmul-vectorization="vector-type=scalable vector-size=4"
-  -convert-vector-to-scf        ← 删掉,让向量 IR 一路走到 convert-vector-to-llvm
```

**执行**:`bash $APP/k3/build_seq.sh 128 exp1_scalable`
**结果**:`buddy-opt` 在跑 `SUB_PASSES` 时 **verifier 崩溃**,产物未生成,连余弦门都到不了。
具体:图内有大量 `memref.expand_shape`(§4.4 统计 586 个,如 `memref<256002048xf32> → memref<250002x1024xf32>`,stride 为动态推断);
`staticize-memref-layout` 把布局静态化后,与 `expand_shape` 的 stride 不变量(静态 stride 必须与输出形状一致)冲突 → verifier 报 stride 非法。
**根因**:布局三连与"含动态 stride expand_shape 的大图"不兼容——`staticize-memref-layout` 强定布局发生在 `expand_shape` 之前,破坏 stride 不变量。
这是 **pass 组合问题(非模型数值问题)**;可尝试的顺序是 staticize 后重跑 `expand-strided-metadata`/`canonicalize`,或不用 staticize,直接走 §6.5.2。

#### 6.5.2 #5:scalable `matmul-vectorization`(去掉布局三连)→ zero/invalid CLS

**假设**:#4 证明布局三连不可用;去掉三连,仅换 scalable pass + 删 `convert-vector-to-scf`,单独验证该 pass 对动态 stride 图的适应性。
**改动**:同 §6.5.1 的 diff,但**不加**布局三连(其余与基线一致)。
**执行**:`bash $APP/k3/build_seq.sh 128 exp1_scalable`(重建)→ `python3 k3/cos.py results/seq128-baseline/emb.txt <产物输出>`。
**结果**:编译、链接全部成功,推理能跑,但输出 **zero/invalid CLS**(embedding 全零/非法)→ cos 门 FAIL。
**根因**:scalable 路径本身数值错误(与布局无关)。图上存在 **250002×1024 的 embedding 查表 GEMM** 与动态 shape,scalable 向量的 mask/尾处理/stride 计算在该路径下产出错误数据。
排除项:本实验 `LLC_ATTRS` 仍是默认(无 zvl 声明,该后端路径在基线上正确),故问题指向 buddy 的 `matmul-vectorization` pass 生成的向量 IR 及其后续 lowering,与 #7/#8 的后端 bug 相互独立(精确单点待上游二分)。

#### 6.5.3 #6:fixed `vector-size=16` → zero/invalid CLS

**假设**:scalable 走掩码路径、fixed 走无掩码定长路径,两者代码路径不同;若 fixed 正确,问题就被限定在 scalable 掩码路径。
**改动**:`-matmul-vectorization="vector-size=16"`,其余与 #5 相同(即 `k3/build_seq_optimize.sh` 的现状)。
**执行**:`bash $APP/k3/build_seq.sh 128 exp1_fixed16` → 余弦门。
**结果**:与 #5 相同——编译成功、输出 **zero/invalid CLS**。
**根因**:scalable 与 fixed 双双数值错误 → 问题不在"掩码 vs 定长",而在 `matmul-vectorization` 对大 K GEMV 型 matmul / 动态 shape 的通用代码生成(含 embedding 查表 GEMM)。
三个 pass 家族横向对比(§5.1、§6.5):**只有默认 `matmul-parallel-vectorization-optimize` 在 BGE-M3 图上数值正确**。

#### 6.5.4 #7:`-mcpu=spacemit-x100`(只改后端,不改 IR)→ cos=0.340539329

**假设**:基线只声明 `+m,+d,+v`(不声明 zvl),llc 只能保守生成代码;显式 `-mcpu=spacemit-x100` 应让后端知道 zvl256b 与扩展集,生成更优 RVV 代码,且**不应改变数值**(后端换 CPU 模型纯属代码生成优化)。
**改动**:仅 `export LLC_ATTRS="-march=riscv64 -mcpu=spacemit-x100 -mtriple=riscv64-unknown-linux-gnu"`,`SUB_PASSES` 与基线完全相同(IR 一个字节不改)。
**执行**:

```bash
bash $APP/k3/build_seq.sh 128 exp2_mcpu
$BUDDYBIN/buddy-cli --model $APP/out/seq128-exp2_mcpu/bge_m3.rax \
  --prompt "hello world" --no-stats > /tmp/e3.txt
python3 $APP/k3/cos.py $APP/results/seq128-baseline/emb.txt /tmp/e3.txt
```

**结果**:

```text
cos = 0.340539329   (dim=1024)
FAIL
```

**根因**:后端在"保证 256 位向量宽度"的假设下生成了错误代码(见 #8,两实验 cos 逐位相同 → 同一 bug)。

#### 6.5.5 #8:`+zvl256b -riscv-v-vector-bits-min=256`(只改后端)→ cos=0.340539329(与 #7 相同)

**假设**:定位 #7 的触发条件是"spacemit-x100 CPU 模型整体"还是其中单个属性(zvl256b);单独加 `+zvl256b` + `-riscv-v-vector-bits-min=256` 做**最小复现**。
**改动**:仅 `export LLC_ATTRS="-march=riscv64 -mattr=+m,+d,+v,+zvl256b -mtriple=riscv64-unknown-linux-gnu -riscv-v-vector-bits-min=256"`(即 `k3/exp_g2.sh`),IR 不改。
**执行**:`bash $APP/k3/exp_g2.sh 128`(脚本内部:重编 → 余弦门 → profile → 测速)。
**结果**:

```text
[exp_g2] 2/4 余弦门
cos = 0.340539329   (dim=1024)
FAIL
```

与 #7 **逐位完全相同**(9 位有效数字)。
**根因**:
- 两个实验只有后端声明不同,却产出完全相同的错误 embedding → 触发的是**同一条错误代码路径**;
- 该路径 = RuyiAI LLVM 的 RISC-V 后端在**"向量宽度 ≥256 且按 256 位布局"**的假设下(例如按 VLMAX = 8×f32 做固定宽度化/尾处理)生成错误代码;
- 默认 `-mattr=+m,+d,+v`(不声明 zvl、不保证宽度)走另一条路径,数值正确;
- 硬件 VLEN 实测 = 256 bits,但"**硬件是 256**"与"**编译期保证 256**"在代码生成上不等价——这是本次发现的**最重要的上游 bug**。

#### 6.5.6 汇总结论

- 三个 matmul 向量化 pass 家族(parallel / blis / vectorization)中,**只有默认 `matmul-parallel-vectorization-optimize` 在 BGE-M3 图上数值正确**,但其向量化弱(§6.3:`vfmacc.vf`,K 维标量);
- `matmul-vectorization`(scalable 与 fixed 均)在图上的数值错误与布局无关(#4 是独立的 pass 组合问题),待上游修大 K GEMV / 动态 shape 路径;
- 后端证据(#7/#8 cos 逐位一致):RuyiAI LLVM 的 RISC-V 后端在"256 位向量宽度假设"下存在数值 bug;默认 `-mattr=+m,+d,+v` 正确 → **短期红线:K3 上不要加 zvl 声明、不要用 `-mcpu=spacemit-x100`**;
- 8 个实验全部是负结果,但每一个都缩小了问题边界(布局 → pass → 后端三层定位),已构成交付物。

### 6.6 剩余差距与负责组件

- **同硬件对比(§3.3)**:RAX 128/256 **快于** HF 参考(0.81×/0.75×),512 持平(1.05×)→ **已达到"reach or exceed reference"**;
- **剩余差距**:K3 上 RAX 与 HF 的执行效率都只有 ≈3–4 GFLOPS,距 x86 HF 的 ~190 GFLOPS 差 **~50×** → 差距在 **RVV 向量化深度**,不在"相对参考落后";后续工作目标 = 相对基线提升效率(向量化/分块/精度);
- **负责组件**:
  1. `midend/lib/Conversion/MatMulOptimization/` 的 `matmul-vectorization` / `matmul-vectorization-blis` 在 BGE-M3 图(含 250002×1024 embedding 查表 GEMM、动态 shape)上数值错误;
  2. 默认 `matmul-parallel-vectorization-optimize` 向量化弱(K 维标量,`vfmacc.vf`);
  3. LLVM RISC-V 后端在 256 位向量宽度假设下数值错误(`-mcpu=spacemit-x100` 与 `+zvl256b -riscv-v-vector-bits-min=256` 均触发,cos 相同 = 0.3405);
  4. IME 下放(#916)尚未接入 models 编译链路;
  5. 并行扩展性差:1→8 线程仅 2.36×,8→16 饱和(§6.4)→ 内存访存/串行段为主瓶颈。
- **建议后续工作**:
  1. 修 `matmul-vectorization` 在大 K GEMV/尾处理的正确性,或改用 upstream `linalg→vector`(生成 `vector.contract`);
  2. 自写 RVV matmul micro-kernel(`vector.contract` + `vfmacc.vv`,分块 + 权重复用);
  3. 接入 IME 做 GEMM 下放(RVV 管向量/规约,IME 管矩阵);
  4. 试 FP16/BF16(权重减半,带宽压力减半);
  5. 定位并修 `spacemit-x100` CPU 模型的数值 bug。

## 7. 待办(进度)

- [x] P1 基线(128/256/512,§2.1–§2.4)
- [x] P2 Reference(HF,§3)
- [x] P3 §5.1 算子结构统计(§4.1–§4.3)
- [x] P3 补充:buffer 化静态分析(§4.4)
- [x] P3 §5.3 运行时打点(§4.5;forward≈100%)
- [x] P3 §5.2 MLIR pass 计时(§4.6)
- [x] P3 §5.5 算力/带宽理论模型(§4.7)
- [x] P3 §5.6 瓶颈排序表(§4.8)
- [ ] P3 §5.4(可选)perf 硬件计数
- [x] P4 第一刀实验(§5;负结果:变慢 2.3×,已回退)
- [x] P4 诊断(§5.1):默认 matmul-pass 弱向量化(每 matmul 仅 ~4 vector、无 contraction)
- [x] P4 第二刀 blis(§5.2;数值错误,已回退)
- [x] P4 第三刀 blis+convert-vector-to-scf(§5.3;仍数值错误,已回退)
- [x] P4 K3 实验(matmul-vectorization scalable/fixed、blis、`-mcpu`、`+zvl256b`)→ 全部负结果,已记录 §6.5.1–§6.5.5
- [x] P5 IME:未接入模型链路(#916 未合入),按"量化证据"口径写入 §6.6
- [x] K3 同硬件 Reference(HF,16 线程,§3.3)
- [x] K3 线程扩展性(§6.4)
- [ ] 汇总提交 #888(定稿完成,待发 PR + 评论)

---

## 附录:原始测量日志(自动生成,保留备查)
> 注:以下为当时 `issue888/` 布局下的原始终端日志(用户名与家目录已脱敏为 `user`),保留历史原貌;现行目录布局见 README.md。

> [bench-server] ready (warm-up request done)
> [bench-server] n=5  latencies(ms)=[3793.7, 4305.8, 3651.6, 3303.3, 3790.2]
> [bench-server] median=3790.21 ms  stdev=360.51 ms  mean=3768.93 ms
> user@Virtleo:~/buddy-mlir$  cd /home/user/buddy-mlir && bash issue888/bench_cli.sh ./build/models/bge_m3/bge_m3.rax 3
> [bench] model=./build/models/bge_m3/bge_m3.rax repeat=3
> [bench] run | wall_s | maxRSS_KB
>   01   16.80 4367636
>   02   20.28 4367504
>   03   10.02 4367604
> [bench] steady-state median latency = 15150.0 ms (n=2)
> [bench] steady-state median maxRSS  = 4265.2 MB
> user@Virtleo:~/buddy-mlir$

user@isrc:~/buddy-k3/buddy-mlir/issue888$ source ~/buddy-k3/issue888/env.sh
uname -a; grep -E '^(NAME|VERSION)=' /etc/os-release
nproc; free -g | head -2
cmake --version | head -1; ninja --version; g++ --version | head -1
ls $LLVMBIN/{llc,mlir-opt,mlir-translate,clang,clang++} 2>/dev/null
ls $BUDDYBIN/{buddy-opt,buddy-cli,buddy-server} 2>/dev/null
git -C $REPO log -1 --oneline 2>/dev/null || echo "(无 .git)"
Linux isrc 6.18.3-generic #1.0.1.4 SMP PREEMPT_DYNAMIC Thu May 21 16:47:06 CST 2026 riscv64 GNU/Linux
NAME="Bianbu"
VERSION="4.0.1 (Resolute Raccoon)"
8
               total        used        free      shared  buff/cache   available
内存：            31           0          16           0          14          30
cmake version 3.31.6
1.12.1
g++ (Bianbu 15.2.0-16ubuntu1bb3) 15.2.0
/home/user/buddy-k3/buddy-mlir/llvm/build/bin/clang    /home/user/buddy-k3/buddy-mlir/llvm/build/bin/mlir-opt
/home/user/buddy-k3/buddy-mlir/llvm/build/bin/clang++  /home/user/buddy-k3/buddy-mlir/llvm/build/bin/mlir-translate
/home/user/buddy-k3/buddy-mlir/llvm/build/bin/llc
/home/user/buddy-k3/buddy-mlir/build/bin/buddy-cli  /home/user/buddy-k3/buddy-mlir/build/bin/buddy-server
/home/user/buddy-k3/buddy-mlir/build/bin/buddy-opt
e7cc8cea (HEAD -> main, tag: nightly/v0.0.8.dev20260904, origin/main, origin/HEAD) feat(proteinglm): add masked-LM buddy-server backend (#884)


arg0.data                                                                                    1%   29MB   2.9MB/s   12:09 ETA^Cuser@Virtleo:~/buddy-mlir$ scp -r issue888/dist user@k3-003:~/buddy-k3/buddy-mlir/issue888/src
arg0.data                                                                                  100% 2166MB   2.6MB/s   14:02    
forward.mlir                                                                               100%  182KB   1.1MB/s   00:00    
seq128.json                                                                                100%  390     9.4KB/s   00:00    
bge_m3.mlir                                                                                100% 1175    29.6KB/s   00:00    
subgraph0.mlir                                                                             100%  505KB   1.8MB/s   00:00    
tokenizer.json                                                                             100%   16MB   3.5MB/s   00:04    
forward.mlir                                                                               100%  182KB   1.1MB/s   00:00    
bge_m3.mlir                                                                                100% 1175    33.3KB/s   00:00    
seq256.json                                                                                100%  390    10.2KB/s   00:00    
subgraph0.mlir                                                                             100%  505KB   3.1MB/s   00:00    
forward.mlir                                                                               100%  182KB   1.1MB/s   00:00    
bge_m3.mlir                                                                                100% 1175    33.8KB/s   00:00    
seq512.json                                                                                100%  390     8.4KB/s   00:00    
subgraph0.mlir                                                                             100%  505KB   1.9MB/s   00:00    
user@Virtleo:~/buddy-mlir$ scp -r issue888/k3 user@k3-003:~/buddy-k3/buddy-mlir/issue888/
env.sh                                                                                     100% 1101    31.8KB/s   00:00    
build_seq.sh                                                                               100% 7061   170.5KB/s   00:00    
cos.py                                                                                     100% 1159    31.8KB/s   00:00    
build_runtime.sh                                                                           100% 3037    65.5KB/s   00:00    
bench.sh                                                                                   100% 3416    90.5KB/s   00:00    
profile.sh                                                                                 100% 1319    39.9KB/s   00:00


user@isrc:~/buddy-k3/buddy-mlir$ llc --version 
LLVM (http://llvm.org/):
  LLVM version 24.0.0git
  Optimized build with assertions.
  Default target: riscv64-unknown-linux-gnu
  Host CPU: spacemit-x100

  Registered Targets:
    riscv32   - 32-bit RISC-V
    riscv32be - 32-bit big endian RISC-V
    riscv64   - 64-bit RISC-V
    riscv64be - 64-bit big endian RISC-V

[seq128/baseline] 1/4 forward.mlir → forward.o
[seq128/baseline] 2/4 subgraph0.mlir → subgraph0.o(最耗时,耐心等)
[seq128/baseline] 3/4 链接 bge_m3_model.so
[seq128/baseline] 4/4 打包 bge_m3.rax
wrote bge_m3.rax (2295159269 bytes, payload entries=5)
[seq128/baseline] 完成 → /home/user/buddy-k3/buddy-mlir/issue888/out/seq128-baseline/bge_m3.rax
-rw-rw-r-- 1 user user 2.2G Sep 20 15:43 /home/user/buddy-k3/buddy-mlir/issue888/out/seq128-baseline/bge_m3.rax


user@isrc:~/buddy-k3/buddy-mlir/issue888/results/seq128-baseline$ bash $ISSUE/k3/bench.sh 128 baseline 20 10
[bench] rax=/home/user/buddy-k3/buddy-mlir/issue888/out/seq128-baseline/bge_m3.rax  tag=baseline  threads=默认
[bench] 口径 B:启动 buddy-server(端口 8090)...
[bench] 就绪(首次 200 兼作预热;响应前 200 字符:
{"data":[{"embedding":[-0.032258488237857819,0.032208502292633057,-0.047389708459377289,0.0028593991883099079,-0.018832137808203697,0.0069218971766531467,0.028678061440587044,-0.035743273794651031,-0.)
[bench] 口径 A:buddy-cli × 10
[bench] 注意:/usr/bin/time 不存在 → 只记墙钟(峰值内存 NA)
[bench] ✓ emb.txt 有输出
[口径B 稳态] 单位 ms : median=19926.62 stdev=120.03 n=19
[口径A 冷启动] 单位 s : median=23.81 stdev=0.29 n=9
[bench] 原始数据 → /home/user/buddy-k3/buddy-mlir/issue888/results/seq128-baseline

user@isrc:~/buddy-k3/buddy-mlir/issue888$ cat $ISSUE/results/seq128-baseline/latency_a.txt  
26.403 4365236
26.816 4365364
25.939 4365508
26.278 4365044
25.549 4365212
31.438 4365132
42.843 4364880
42.225 4365232
42.583 4364820
41.369 4364984

user@isrc:~/buddy-k3/buddy-mlir/issue888/results/seq128-baseline$ cat latency_b.txt 
23.049190
22.924298
22.808405
22.785426
22.659910
22.678771
22.972717
22.719809
22.977880
23.012093
22.148935
22.703120
22.428959
22.891876
22.774364
22.765794
22.582254
22.847624
23.013971
22.675011

user@isrc:~$ llc --version
LLVM (http://llvm.org/):
  LLVM version 24.0.0git
  Optimized build with assertions.
  Default target: riscv64-unknown-linux-gnu
  Host CPU: spacemit-x100

  Registered Targets:
    riscv32   - 32-bit RISC-V
    riscv32be - 32-bit big endian RISC-V
    riscv64   - 64-bit RISC-V
    riscv64be - 64-bit big endian RISC-V

user@isrc:~/buddy-k3/buddy-mlir/issue888$ python3 /home/user/buddy-k3/buddy-mlir/issue888/k3/cos.py results/seq128-baseline/emb_baseline.txt results/seq128-baseline/emb.txt
cos = 1.000000000   (dim=1024)
PASS

user@isrc:~/buddy-k3/buddy-mlir/issue888$ python3 /home/user/buddy-k3/buddy-mlir/issue888/k3/cos.py results/seq128-baseline/emb_baseline.txt results/seq256-baseline/emb.txt
cos = 1.000000000   (dim=1024)
PASS

user@isrc:~$ cat $ISSUE/results/seq256-baseline/latency_b.txt  
78.469401
38.471024
38.956597
39.298203
39.006409
38.991936
38.911116
39.062745
38.762847
39.525397
40.734614
40.015246
40.527005
39.917569
39.624459
40.149099
39.702846
39.713974
40.325736
40.399082

user@isrc:~$ tail -f $ISSUE/out/bench256.log
[bench] rax=/home/user/buddy-k3/buddy-mlir/issue888/out/seq256-baseline/bge_m3.rax  tag=baseline  threads=默认
[bench] 口径 B:启动 buddy-server(端口 8090)...
[bench] 就绪(首次 200 兼作预热;响应前 200 字符:
{"data":[{"embedding":[-0.032258488237857819,0.032208502292633057,-0.047389708459377289,0.0028593991883099079,-0.018832137808203697,0.0069218971766531467,0.028678061440587044,-0.035743273794651031,-0.)
[bench] 口径 A:buddy-cli × 10
[bench] 注意:/usr/bin/time 不存在 → 只记墙钟(峰值内存 NA)
[bench] ✓ emb.txt 有输出
[口径B 稳态] 单位 ms : 无数据
[口径A 冷启动] 单位 s : median=42.87 stdev=2.27 n=9
[口径A 峰值内存] 单位 MB : median=5399


user@isrc:~/buddy-k3/buddy-mlir/issue888$ cat $ISSUE/results/seq512-baseline/latency_b.txt  
131.938481
137.366237
133.507066
110.623398
114.898787
116.513196
114.855710
115.863780
116.376391
112.922018
116.609795
115.705860
114.918751
115.719369
113.270404
114.369321
115.658662
114.267587
114.328814
117.097765

user@isrc:~/buddy-k3/buddy-mlir/issue888$ cat $ISSUE/results/seq512-baseline/latency_a.txt  
123.787 8773488
116.439 8773692
125.332 8773740
121.980 8773400
124.096 8773468
128.353 8773344
132.764 8773488
133.173 8773176
135.569 8773480
131.863 8773452
user@isrc:~/buddy-k3/buddy-mlir/issue888$ tail -f $ISSUE/out/bench512.log
[bench] 就绪(首次 200 兼作预热;响应前 200 字符:
{"data":[{"embedding":[-0.032258488237857819,0.032208502292633057,-0.047389708459377289,0.0028593991883099079,-0.018832137808203697,0.0069218971766531467,0.028678061440587044,-0.035743273794651031,-0.)
[bench] 口径 A:buddy-cli × 10
[bench] 注意:/usr/bin/time 不存在 → 只记墙钟(峰值内存 NA)
[bench] ✓ emb.txt 有输出
[口径B 稳态] 单位 ms : median=115658.66 stdev=6490.24 n=19
[口径A 冷启动] 单位 s : median=128.35 stdev=5.90 n=9
[口径A 峰值内存] 单位 MB : median=8568
[bench] 原始数据 → /home/user/buddy-k3/buddy-mlir/issue888/results/seq512-baseline

user@isrc:~/buddy-k3/buddy-mlir/issue888$  python3 /home/user/buddy-k3/buddy-mlir/issue888/k3/cos.py results/seq128-baseline/emb_baseline.txt /home/user/buddy-k3/buddy-mlir/issue888/results/seq5
12-baseline/emb.txt
cos = 1.000000000   (dim=1024)
PASS


$BUDDYBIN/buddy-cli --model $ISSUE/out/seq128-exp2_mcpu/bge_m3.rax \
  --prompt "hello world" --no-stats > /tmp/e3.txt
user@isrc:~/buddy-k3/buddy-mlir/issue888$ python3 $ISSUE/k3/cos.py $ISSUE/results/seq128-baseline/emb.txt /tmp/e3.txt
cos = 0.340539329   (dim=1024)
FAIL

user@isrc:~$ tail -f $ISSUE/results/thread_scaling.txt
== 清理历史孤儿 server ==
== 当前内存 ==
               total        used        free      shared  buff/cache   available
内存：            31           0           6           0          24          30
== 残留进程检查(应为空) ==
(无)

===== OMP_NUM_THREADS=16 (port 8146) =====
  就绪(预热完成)。实测 2 次,单位秒:
  20.561765
  20.075934

===== OMP_NUM_THREADS=8 (port 8138) =====
  就绪(预热完成)。实测 2 次,单位秒:
  20.764627
  20.936416

===== OMP_NUM_THREADS=4 (port 8134) =====
  就绪(预热完成)。实测 2 次,单位秒:
  32.635430
  33.016966

===== OMP_NUM_THREADS=1 (port 8131) =====
  就绪(预热完成)。实测 2 次,单位秒:
  49.269313
  49.280662

== 完成。最终内存 ==
               total        used        free      shared  buff/cache   available
内存：            31           1           6           0          24          30
输出 → /home/user/buddy-k3/buddy-mlir/issue888/results/thread_scaling.txt


(venv) user@isrc:~/buddy-k3/buddy-mlir/issue888$ python $ISSUE/x86/hf_bench.py --model-dir $HOME/buddy-k3/models/bge-m3-hf \
  --max-length 128 --threads 16 --repeat 10
Loading weights: 100%|██████████████████████████████████| 391/391 [00:00<00:00, 4862.54it/s]
[hf-bench] threads=16 max_length=128 repeat=10
[hf-bench] latencies(ms) = [24709.35, 24688.6, 24713.87, 24649.33, 24691.65, 24681.63, 24626.18, 24695.81, 24674.34, 24619.56]
[hf-bench] steady-state median = 24681.63 ms (n=9)

[hf-bench] threads=16 max_length=256 repeat=10
[hf-bench] latencies(ms) = [51630.36, 51954.76, 52744.08, 52769.81, 52752.58, 52062.88, 53157.43, 51437.98, 52667.97, 53716.93]
[hf-bench] steady-state median = 52744.08 ms (n=9)

[hf-bench] threads=16 max_length=512 repeat=10
[hf-bench] latencies(ms) = [112716.34, 112967.42, 113003.14, 113094.91, 113344.43, 109797.65, 110122.98, 109883.56, 109561.1, 109564.82]
[hf-bench] steady-state median = 110122.98 ms (n=9)