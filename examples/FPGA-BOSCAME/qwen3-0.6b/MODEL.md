# Qwen3-0.6B 算子与形状

本目录逐算子验证 Qwen3-0.6B 的模型语义和 BOSCAME NR 部署路径。它不加载模型权重，不执行整模型文本生成。测试数据可复现，每个 `launch.c` 调用实际从 `kernel.mlir` 降低得到的函数，并逐元素比较独立 C 参考结果。

结构来源是 [michaelcjl/qwen3-0.6b](https://gitlink.org.cn/michaelcjl/qwen3-0.6b.git) 的 `e9bb2dc3e2fe3893cd84f8bc19d0872618c8a0d6` 版本，并与 [官方 Qwen/Qwen3-0.6B config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) 核对。旧仓库中的 v0.1 算子和板级实现不作为实现来源。NR 指令、启动、缓存同步、链接和镜像生成依据 [ModelZoo](https://gitlink.org.cn/michaelcjl/ModelZoo.git) 的 `8815b74fb6d3cd6288c4d99ac6fd7c5d041a7cc3` 版本。构建本目录不依赖这两个参考仓库。

机器可读的参数、形状和语义说明见 [model.json](model.json)；具体算子的缓冲区形状、类型和测试范围见各目录的 `metadata.json`。

结构核对位置如下，行号针对上述锁定版本：

| 参考文件 | 行号 | 核对内容 |
| --- | --- | --- |
| `config.json` | 9–29 | 官方模型维度、层数、RoPE、归一化、词表和 dtype |
| `convert_qwen3_06b_fp32.py` | 63–101 | 从实际参数形状读取 Q/K/V/O 维度；强制存在 Q/K norm |
| `convert_qwen3_06b_fp32.py` | 168–194 | 每层参数、gate/up/down、最终 norm 和 lm_head |
| `qwen3_run.c` | 338–433 | 层级算子调用顺序和残差、KV/GQA 数据流 |
| `qwen3_run.c` | 607–648 | prompt 长度可变；prefill 与 decode 均逐 token 调用 |

参考仓库把 Q/K/V 权重沿输出维拼成 `[4096,1024]`，这是三个线性投影的存储/执行融合。本目录按照官方的三个投影语义拆开，因此 K/V 共用一个 `[M,1024,1024]` 形状，不需要把融合后的 `[M,4096,1024]` 再计为模型必须拥有的新算子。

仓库附带的历史 `qwen3-0.6b-fp32.bin` 是 header version 1，其 RoPE theta 为 10000，且没有 v2 的 Q/K norm 参数标志；它不能替代当前 `config.json` 和导出脚本的结构定义。这里使用官方 `theta=1000000` 和 Q/K norm。[Transformers v4.51.0 的 Qwen3 定义](https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen3/modeling_qwen3.py#L98) 同时确认 RoPE 使用 split-half 排布。

## 结构参数

| 参数 | 数值 | 含义 |
| --- | ---: | --- |
| batch | 1 | 本目录固定批量 |
| hidden_size | 1024 | 残差和词嵌入宽度 |
| intermediate_size | 3072 | SwiGLU 中间宽度 |
| num_hidden_layers | 28 | 每层复用相同算子形状 |
| num_attention_heads | 16 | Q 头数 |
| num_key_value_heads | 8 | K/V 头数，GQA 重复因子为 2 |
| head_dim | 128 | **由配置显式指定，不能用 1024 / 16 推导** |
| Q projection width | 2048 | 16 × 128 |
| K/V projection width | 1024 | 8 × 128，各一个投影 |
| vocab_size | 151936 | 完整词表 |
| rms_norm_eps | 1e-6 | 输入、MLP、最终和 Q/K 归一化 |
| rope_theta | 1000000 | 完整 128 维 RoPE，split-half 排布 |
| max_position_embeddings | 40960 | 官方默认位置上限 |
| tie_word_embeddings | true | lm_head 与 embedding 共享权重 |
| attention_bias | false | 线性投影无 bias 加法 |
| use_sliding_window | false | 使用全因果注意力 |
| checkpoint dtype | BF16 | 算子测试另外明确 F32 和 W8A8 路径 |

## 一个 decoder 层的计算

令 `S` 为当前输入 token 数，`P` 为已有缓存长度，`T=P+S`。

1. 对 `X[S,1024]` 执行 RMSNorm。
2. Q/K/V 三个线性投影分别得到 `[S,2048]`、`[S,1024]`、`[S,1024]`，视图变换为 `[S,16,128]`、`[S,8,128]`、`[S,8,128]`。
3. Q 和 K 各自沿 128 维执行带独立权重的 RMSNorm，然后按绝对位置 `P+s` 施加 RoPE。V 不执行 Q/K norm 和 RoPE。
4. 把新的 K/V 写入缓存；为每两个 Q 头选取同一个 KV 头。GQA 对应 `kv_head=floor(q_head/2)`，不是交替取头。
5. 按头计算 `Q × Kᵀ`，得到 `[16,S,T]`，乘 `1/sqrt(128)`，对 `t>P+s` 的位置施加负无穷 mask，沿最后一维执行稳定 softmax。
6. 计算 `probabilities × V`，得到 `[16,S,128]`；交换头和 token 维并拼成 `[S,2048]`，经 O 投影得到 `[S,1024]`，与层输入残差相加。
7. 执行第二个 RMSNorm；分别计算 gate/up 投影 `[S,3072]`；计算 `SiLU(gate) * up`；经 down 投影回到 `[S,1024]`，再加残差。

上述层重复 28 次。随后执行最终 RMSNorm，并将所需 token 的 `[M,1024]` 乘共享词嵌入权重的转置，得到 `[M,151936]` logits。生成下一个 token 只需要最后一个位置，因此这里的 lm_head 基准使用 `M=1`。训练或返回所有位置 logits 时可以使用 `M=S`；它不增加新的算子种类。

## 静态样例覆盖

序列长度和缓存长度会变化，无法穷举所有 `S/T` 形状。本目录选择两个明确的推理时刻：

| profile | S | P | T | 含义 |
| --- | ---: | ---: | ---: | --- |
| prefill | 16 | 0 | 16 | 16 个 token 的首次填充 |
| decode | 1 | 16 | 17 | 已缓存 16 个 token 后再输入一个 |

独立 KV 更新样例使用 `[8,32,128]` 缓存容量，以便同时检查新增内容和未写入区域保持不变；32 是测试容量，不是模型上下文限制。RoPE 测试从非零绝对位置 7 开始，以发现把所有位置误用为 0 的实现错误；sin/cos 作为预计算输入。模型公式为 `inv_freq[k]=theta^(-2k/128)`，`angle=position*inv_freq[k]`。

目录名 `matmul_MxNxK` 使用数学约定 `A[M,K] × B[K,N] → C[M,N]`，不是按权重的存储顺序命名。

| 模型用途 | M | N | K | 每层出现次数 |
| --- | ---: | ---: | ---: | ---: |
| q_proj | 16 / 1 | 2048 | 1024 | 1 |
| k_proj / v_proj | 16 / 1 | 1024 | 1024 | 2 |
| o_proj | 16 / 1 | 1024 | 2048 | 1 |
| gate_proj / up_proj | 16 / 1 | 3072 | 1024 | 2 |
| down_proj | 16 / 1 | 1024 | 3072 | 1 |
| lm_head | 1 | 151936 | 1024 | 模型末尾 1 次 |

QK 和 PV 保持 F32：

| 操作 | prefill | decode |
| --- | --- | --- |
| QK，16 个头 | `[16,16,128] × [16,128,16]` | `[16,1,128] × [16,128,17]` |
| PV，16 个头 | `[16,16,16] × [16,16,128]` | `[16,1,17] × [16,17,128]` |

其他实际计算全部通过 `linalg.generic` / `linalg.fill` 表达：

| 操作 | 实例形状与语义 |
| --- | --- |
| embedding | `[151936,1024]` 词表，gather 16 / 1 个 token；覆盖词表最后一行 |
| RMSNorm | `[16,1024]`、`[1,1024]`；Q 的 `[256,128]` / `[16,128]`；K 的 `[128,128]` / `[8,128]` |
| RoPE | `[16,16,128]`、`[16,8,128]`、`[1,16,128]`、`[1,8,128]`；内部视图 `[S,H,2,64]` |
| scale + causal mask | `[16,16,16]`、`[16,1,17]` |
| stable softmax | 同上，先减最大值，再求指数与行和 |
| SiLU / 逐元素乘法 | `[16,3072]`、`[1,3072]` |
| residual add | `[16,1024]`、`[1,1024]` |
| KV cache update | `[S,8,128]` 写入 `[8,32,128]`；K 和 V 复用同种操作 |
| GQA repeat | `[8,T,128] → [16,T,128]`，T=16 / 17 |
| Q layout | `[S,16,128] → [16,S,128]` |
| K transpose | `[16,T,128] → [16,128,T]` |
| context layout | `[16,S,128] → [S,16,128]`，随后无数据搬运地视图变换为 `[S,2048]` |

纯 reshape、切片视图、选取最后一个 token 和共享权重可由 memref 描述符表示，不额外生成算术内核。tokenizer 和采样策略属于文本生成外围流程；它们不属于本目录的 decoder 算子集合。

## 精度与部署边界

官方 checkpoint 为 BF16。这里的 F32 样例验证相同的算子公式和实际模型维度，避免把 BF16 每层舍入误差与降低或硬件错误混合；它们不声称逐比特复现官方 BF16 整模型输出。

BOSCAME NR 整数线性路径采用有符号 W8A8 输入与 I32 累加。对应的逐 token 量化和逐行、逐列 scale 反量化也分别作为 linalg 算子验证。这些是显式的部署变换，**不是官方模型结构自带的整数算子，也不等价于 BF16 权重**。无权重的独立样例不能证明量化后的整模型文本质量。

`matmul_3x19x70` 用于 NR tile 尾部和 K 分块回归，明确不属于 Qwen3 的模型形状。28 层只重复同类算子，不需要复制 28 份相同目录。
