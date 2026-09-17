# 外部调用返回描述符的契约（R1 结论 + 本地核验）

本文记录 `model/` 把图算子替换成 Triton 静态库调用时踩到的一个**契约错误**，以及它的
证据和修法。结论经过：(a) 外部评审者 R1 的解答，(b) 本仓库源码逐条核验，(c) 主机数值
验证。

## 现象

替换后的图能编译、能链接、能进到 kernel，但在 `triton_matmul_*` 里段错误。gdb 调用链：

```text
_mlir__mlir_ciface_forward_prefill          (JIT 入口)
  → _mlir_ciface_forward_prefill            (MLIR C wrapper)
    → forward_prefill                       (图 body)
      → _mlir_ciface_kernel_matmul_16x2048x1024_f32   (archive per-case 适配层)
        → triton_matmul_16x2048x1024_f32    (编译出的 Triton kernel) ← SIGSEGV
```

`forward_prefill` 的 body 用「按值展开的 memref 描述符 + 返回 descriptor struct」调用外部
符号，一度被误判为 ABI 不匹配。

## 三条被核验的事实（全部来自本仓库源码）

### 1. 外部声明的桥接方向

`llvm/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp:425-435`：

```cpp
static void wrapWithCInterface(...) {
  if (newFuncOp.isExternal())
    wrapExternalFunction(...);     // 外部函数
  else
    wrapForExternalCallers(...);   // 有 body 的函数
}
```

`:188-200` 创建 `_mlir_ciface_<name>`（`Linkage::External`），并把**原函数设为 private 并
添加 entry block**（即给它一个自动生成的 body）。

所以外部函数 `@foo` 的形态是：

```text
@foo                      自动生成的 private 桥接函数，有 body
  → @_mlir_ciface_foo     外部声明，由 C 库提供实现
```

用 MLIR Python API 遍历真实模块，读数与之一致：

```text
qwen_graph_matmul_16x2048x1024_f32               body=True   calls=['_mlir_ciface_qwen_graph_matmul_16x2048x1024_f32']
_mlir_ciface_qwen_graph_matmul_16x2048x1024_f32  body=False  calls=[]
```

因此「body 里出现 RAW 调用」是**正常转换结果**，不是接口错误；C 库只要导出
`_mlir_ciface_<symbol>` 即可。

> 教训：早期用「grep 调用行 + 最近一次 `llvm.func`」的文本扫描判断外层函数，把
> `_mlir_ciface_case_a` 的 body 里的调用算到了 `case_a` 头上，据此得出错误结论。
> 必须用 MLIR API 按 op 的父函数归属，或至少区分声明与定义。

### 2. ciface 的第一个指针是**未初始化的返回描述符槽**

`FuncToLLVM.cpp:205-216`：

```cpp
if (resultStructType) {
  // Allocate the struct on the stack and pass the pointer.
  Value result = LLVM::AllocaOp::create(builder, loc, resultType, resultStructType, one);
  args.push_back(result);
}
```

只 `alloca` 了 descriptor 结构体，**没有初始化它的数据指针，也没有分配数据**。

所以「返回 memref」的外部 callee **必须自己把结果描述符填成合法状态**。旧版适配层直接
读 `a0->aligned` 当输出缓冲区用 —— 读的是调用方栈上的垃圾，这正是段错误的来源。

### 3. 仓库既有的 RNG 外部算子是**分配**的

`runtime/external_kernels/rng/RNGUtils.cpp:105-117`：

```cpp
// The caller provides an uninitialized result memref descriptor via the
// C-interface wrapper. Allocate storage here and fill the descriptor.
void *buf = nullptr;
...
if (posix_memalign(&buf, 64, ((bytes + 63) / 64) * 64) != 0)
  buf = std::malloc(bytes);
out->basePtr = static_cast<float *>(buf);
out->data = static_cast<float *>(buf);
```

即这条例约**本来就是要 callee 分配**的。提问时我写的「该约定不自行分配输出」是错的。

## 修法

`tools/triton_call_replace.py` 生成的适配层现在做三件事（仍**不含任何数值计算**）：

1. 把结果描述符填成指向**本符号专属的静态缓冲区**（板上即规划好的静态 workspace）：

   ```c
   static float qwen_out_qwen_graph_matmul_16x1024x1024_f32[16384];

   void _mlir_ciface_qwen_graph_matmul_16x1024x1024_f32(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {
     float *out = qwen_out_qwen_graph_matmul_16x1024x1024_f32;
     for (int i = 0; i < 16384; ++i) out[i] = 0.0f;   /* linear 是累加语义 */
     a0->allocated = (void *)0;                        /* 非拥有：见下 */
     a0->aligned = out;
     a0->offset = 0; a0->sizes[0] = 16; a0->strides[0] = 1024;
     a0->sizes[1] = 1024; a0->strides[1] = 1;
     _mlir_ciface_kernel_matmul_16x1024x1024_f32((MemRef2 *)a1, (MemRef2 *)a2, (MemRef2 *)a0);
   }
   ```

2. **`allocated` 置 NULL**。这一步是必须的：图的 `forward_prefill` 里有 101 个 `free`
   调用（`ownership-based-buffer-deallocation` 插入），它会对调用结果的
   `descriptor.allocated` 调 `free()`。指向静态数组会以
   `munmap_chunk(): invalid pointer` 中止。置 NULL 表示「这是调用方不拥有的
   workspace 视图」，与事实一致。

3. 每次调用前清零输出缓冲区 —— 这是 `linear` kernel 的 `accumulator + previous`
   累加语义要求的 accumulator 初始化。原图本来也会对 matmul 目的地做零初始化
   （`outs(%cst)`，`%cst = arith.constant dense<0.0>`），所以这不是额外开销。

静态输出 workspace 合计 **360,832 个 f32（1.38 MiB）**，按符号各一份。

## 已知假设与它的检验方式

**一个符号一份缓冲区** ⇒ 若同一符号的两个结果同时存活，就会互相覆盖。这个假设**不靠
推理、靠数值检验**：替换后的图跑出来的结果与 FP32 参考逐 token 比对，若别名发生，
logits 会明显错开。见 `validation/stage-cd/replaced-28l-host.json`。

## 仍未做

- 上板：静态 workspace 目前是适配层里的数组，还没有和 `common/nr/nr.ld` 的
  `.workspace` 段、以及 `validation/weight-image.json` 的 DDR 布局统一规划。
- 更彻底的形态是评审者建议的「输出作为显式 memref 参数、函数返回 void」，
  在 memref 层重写调用点；那样连返回描述符都不需要。本文件记录的是当前**已验证可用**
  的最小修法。
- attention / RoPE / SiLU / embedding / KV 写入的匹配规则仍未实现。
