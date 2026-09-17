结论：当前工作区没有复现出 A/B 所描述的 ABI 选择差异。按你给的 /tmp/extcall_test.mlir 和命令实测，case_a 的 body 也是 RAW 调用；CIFACE 调用位于自动生成的 ext_returning 桥接函数内。更直接的问题是输出存储约定不匹配：返回 memref 时，ciface 的第一个指针指向“未初始化的返回描述符槽”，不是“包含已分配输出数据地址的描述符”。

你引用的 RNG 实现实际上会分配输出数据。若要求外部 kernel 不分配输出，建议采用 Q2 的显式 out-memref 参数 + void 返回方案，保留编译器生成的 RAW → CIFACE 桥接。

以下仅依据当前工作区源码和只读复现实验；没有运行构建、JIT 模型或 FPGA，没有修改源码。

**Q1：具体条件是什么？首先校正最小复现的函数归属。**

我实际运行了你提供的完整 pipeline，逐个记录两空格缩进的 llvm.func 顶层函数头，再归属其中的 llvm.call。相关调用关系如下：

| 真正的外层函数 | 调用目标 | 形式 |
|---|---|---|
| ext_returning | _mlir_ciface_ext_returning | 3 个 descriptor 指针，void 返回 |
| case_a | ext_returning | 14 个展开参数，返回 descriptor struct |
| _mlir_ciface_case_a | case_a | 14 个展开参数，返回 descriptor struct |
| ext_void | _mlir_ciface_ext_void | 3 个 descriptor 指针，void 返回 |
| case_b | ext_void | 21 个展开参数，void 返回 |
| _mlir_ciface_case_b | case_b | 14 个展开参数，返回 descriptor struct |

特别是：

- CIFACE ext_returning 调用不在 case_a 内。
- _mlir_ciface_case_a 调的是 case_a，不是 ext_returning。
- ext_returning 在输入中无 body，但转换后拥有编译器生成的 private body。
- _mlir_ciface_ext_returning 转换后只有声明，留给 C 库提供实现。

可直接重跑，全部输出到终端，不生成文件：

~~~bash
cd /home/chh/gitprojects/buddy-mlir-for-fpga

build-migrate/bin/buddy-opt /tmp/extcall_test.mlir \
  --pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries},expand-strided-metadata,finalize-memref-to-llvm,func.func(llvm-request-c-wrappers),convert-func-to-llvm,reconcile-unrealized-casts)' \
  | awk '
      /^  llvm.func / {
        header = $0
        sub(/\(.*/, "", header)
        print header
      }
      /llvm.call @/ { print }
    '
~~~

除了上表还会看到 malloc、memrefCopy，属于这份例子的 bufferization 结果，不影响上述 wrapper 方向。

本次核验对象：

~~~text
LLVM checkout:
6ceae32b77a2c0cc3c929ae8924254525155a26c

buddy-opt SHA256:
4b0fb08203af12c96e9dc0995842abb396b6c7a7a4f4752cd04bf52d4710360f

/tmp/extcall_test.mlir SHA256:
721ceda0607dc37991532f7c75d3e3ef65141bc94fdc95f1dbc538674f8eea25
~~~

源码给出的决定条件很明确：

1. llvm-request-c-wrappers 只给它遍历到的 func.func 设置 llvm.emit_c_interface unit attribute。

   证据：llvm/mlir/lib/Dialect/LLVMIR/Transforms/RequestCWrappers.cpp:27。

2. convert-func-to-llvm 检查函数是否带这个 unit attribute，且没有启用 bare-pointer calling convention，才生成 C interface。

   证据：llvm/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp:471、509；bare-pointer 判定在 :58。

3. 原函数没有 body 时，生成方向是：

~~~text
模型 body
  → RAW @foo（自动生成的 private 桥接函数，有 body）
  → @_mlir_ciface_foo（外部声明，C 库提供）
~~~

   证据：同文件 :427–436 的 isExternal 分支；:192–200 创建 ciface 声明并给 RAW 函数添加 private body；:255 调 ciface。

4. 原函数已有 body 时，生成方向是：

~~~text
外部 C 调用方
  → @_mlir_ciface_foo（自动生成的 wrapper，有 body）
  → RAW @foo（原函数转换得到的实现）
~~~

   证据：同文件 :114–162，尤其 :125、154。

5. 普通 func.call 的 lowering 保留原 callee 名字，不因为 llvm.emit_c_interface 自动改成 _mlir_ciface_ 前缀。

   证据：同文件 :600–608，直接执行 newOp.setCalleeAttr(callOp.getCalleeAttr())。:684–703 查询 callee 的分支检查的是 llvm.bareptr，不是 emit_c_interface。

这也与 llvm/mlir/docs/TargetLLVMIR.md:614–641 的两种 wrapper 描述一致。

因此，你列举的 module 函数数量、Python func.FuncOp 构造方式、bufferization 是否 out-of-place，并不是“普通调用改成 CIFACE / 保留 RAW”的分支条件。声明若在 request pass 之后才创建且没有显式属性，会影响是否生成桥接；有无 body 会影响桥接方向。它们都不意味着普通 func.call 会自动改叫 ciface。

对于 B：本次没有找到并重新解析你统计时所用的完整最终 LLVM dialect dump，因此那张表的精确计数和每个调用归属，我不确定。不能据此声称已复现 B 的全部细节。但 forward_prefill 中存在 RAW 调用本身完全符合正常转换结果。应检查对应 RAW qwen_graph_* 是否已有 private body，再看它是否调用外部 ciface 声明。

尤其是“_mlir_ciface_qwen_graph_* 内又出现 CIFACE 调用”这一归属值得重新核验：对于初始无 body 的外部函数，正常生成的 ciface 是声明，本来没有内部调用。建议用 MLIR operation 的父函数做统计；若用文本扫描，必须识别 llvm.func private @foo 等所有函数头，并区分声明与定义。仅 grep 调用行不足以判定外层函数。

“JIT 缺少 ciface 符号，而模型 body 调 RAW”也不矛盾：JIT 会解析 RAW 桥接内部引用的 ciface 符号，C 库无需另外导出 RAW 名字。

**导致段错误的关键契约差异：返回描述符槽 ≠ 已分配输出 buffer。**

对于：

~~~mlir
func.func private @foo(memref<...>, memref<...>) -> memref<...>
    attributes {llvm.emit_c_interface}
~~~

生成的 RAW 桥接逻辑相当于：

~~~cpp
// 伪代码，仅说明语义：
Descriptor result;                 // 未初始化！
Descriptor a = pack(raw_a_args);    // 已初始化
Descriptor b = pack(raw_b_args);    // 已初始化
_mlir_ciface_foo(&result, &a, &b);
return result;
~~~

FuncToLLVM.cpp:208–216 只 alloca 结果 descriptor，没有初始化其数据指针，也没有分配 tensor 数据；:257–260 在外部函数返回后 load 该 descriptor。

因此 ciface 必须先构造合法的结果描述符，包括分配/提供有效数据存储，再让调用者读取。返回合法 alias 也是另一种需要明确所有权的契约，并非所有 returning-memref 都必然要求 malloc；但这里绝不能假定 result.data 预先有效。

你的 RNG 引用在当前源码中的真实链路为：

~~~text
RNGUtils.cpp:729–731
  _mlir_ciface_buddy_bernoulli_f32_r1_1
    → bernoulliF32<1>

RNGUtils.cpp:121–123
  bernoulliF32
    → initOutLike

RNGUtils.cpp:105–117
  明确说明 caller 提供未初始化 result descriptor；
  posix_memalign / malloc 分配数据；
  设置 out->basePtr / out->data。
~~~

所以“该既有约定不自行分配输出”这一前提与当前实现不符。

还发现与你当前适配代码直接对应的证据：

- examples/FPGA-BOSCAME/qwen3-0.6b/model/build/triton-call/full-28l/qwen_triton_adapters.c:34–35，直接把首个 a0 描述符交给 Triton adapter 作为输出。
- examples/FPGA-BOSCAME/qwen3-0.6b/triton/build/matmul_16x1024x1024_f32/adapter.c:9，立即读取输出描述符的 aligned 和 offset 来计算地址。

若调用端仍是 returning-memref 契约，上述链路会读取未初始化结果描述符，这是可以从代码确认的接口错误。是否就是你那一次段错误的实际触发点，本次没有崩溃栈/运行时地址证据，不确定；不能把静态定位冒充动态确认。

**Q2：要求 kernel 不分配输出时，采用显式 out 参数，返回 void。**

建议在 bufferization 后、finalize-memref-to-llvm 和 ownership-based-buffer-deallocation 之前，原子地改写外部声明和调用点。例如：

~~~mlir
module {
  // 约定顺序：out, a, b。没有函数结果。
  func.func private @ext_out(
      memref<4x8xf32>,
      memref<4x8xf32>,
      memref<8x8xf32>)
      attributes {llvm.emit_c_interface}

  // 输出数据存储由上层调用方或 workspace 规划提供。
  func.func @caller(
      %out: memref<4x8xf32>,
      %a: memref<4x8xf32>,
      %b: memref<8x8xf32>) {
    func.call @ext_out(%out, %a, %b)
        : (memref<4x8xf32>, memref<4x8xf32>,
           memref<8x8xf32>) -> ()
    return
  }
}
~~~

对应 C++ 导出：

~~~cpp
extern "C" void _mlir_ciface_ext_out(
    StridedMemRefType<float, 2> *out,
    StridedMemRefType<float, 2> *a,
    StridedMemRefType<float, 2> *b);
~~~

这里 out 是函数的显式第一个 memref 参数，因此 wrapper 会像处理其他输入一样，先 pack 完整描述符、store 到栈槽，再把指针传给 C。kernel 只写 out 指向的数据区域，不负责分配，也不需要替换描述符。

我已用现有 buddy-opt 对上面这个例子运行：

~~~text
builtin.module(finalize-memref-to-llvm,
               convert-func-to-llvm,
               reconcile-unrealized-casts)
~~~

实测读数：

~~~text
caller：
  llvm.call @ext_out(21 个展开参数) -> ()

自动生成的 private ext_out：
  3 个 descriptor alloca
  3 个对应的 descriptor store
  llvm.call @_mlir_ciface_ext_out(ptr, ptr, ptr) -> ()
  llvm.return

外部声明：
  llvm.func @_mlir_ciface_ext_out(!llvm.ptr, !llvm.ptr, !llvm.ptr)
~~~

这是满足“外部 C 库接收 out_ptr/in_ptr，且 kernel 不分配输出”的正确结果。不要把模型 body 仍调用 RAW bridge 当成失败标准。

真实图中的改写还必须做这些事情：

1. 为原调用结果提供合法数据存储，例如计划好的 workspace 子视图；主机原型也可以由 caller 的 memref.alloc 提供。
2. 将该输出 memref 放到调用的第一个 operand。
3. 删除调用的 memref result，把原 result 的所有 uses 重定向到这个输出 memref；有多结果时逐项处理。
4. 同步更新 callee 类型及所有调用点，保留正确的实际 offset、strides、shape 和 alias 语义。
5. 让后续所有权/释放分析基于改写后的 IR 运行；静态 workspace 不得被当成需要 free 的堆分配。
6. 若 kernel 是累加语义，还须由明确的一方初始化 accumulator。ABI 改写不会自动清零数据。

例子采用连续 identity-layout memref 仅为简洁。真实图若有动态 stride/offset，应保留可表达它们的 memref 类型并正确转换 descriptor，不能凭 shape 相同就假设连续。

不要只把 tensor-returning 声明改成 void，然后继续把 tensor.empty 当成被外部调用原地改写的 tensor 结果。tensor 层有不可变 SSA/alias 分析约束：若选择在 tensor 层完成这件事，需要正确的 destination-style 和 BufferizableOpInterface 语义；仅改签名不能建立这些语义。最小风险的处理点是仍保留结构化 memref 的阶段。

还有一个本 checkout 特有、可直接核验的限制：不能直接加 buffer-results-to-out-params 就期待它完成这次外部 ABI 改写。

- llvm/mlir/lib/Dialect/Bufferization/Transforms/BufferResultsToOutParams.cpp:321–322 明确跳过无 body 的外部函数。
- 同文件 :241–242 跳过这些外部函数的调用。
- 对该 pass 支持的函数，输出参数也追加在末尾，不是你需要的 out-first；见 llvm/mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.td:327。

如果你字面上要求“forward_prefill 的 LLVM body 必须直接调用三指针符号”，默认 convert-func-to-llvm 没有一个 emit_c_interface 属性可以实现这个重定向。可另做显式 LLVM descriptor packing + pointer-call lowering，或在后续 LLVM 优化中内联 private bridge，但这与输出存储契约是两件事，且没有必要为正确链接强求直接调用。

llvm.bareptr 也不是解决办法：它表示数据指针 ABI，不能当作 StridedMemRefType* 的描述符指针 ABI 使用。

**Q3：RAW 调用的精确展开顺序和返回类型。**

在你给出的 f32、rank=2、index 降为 i64、默认地址空间的场景，每个 memref 展开为：

~~~text
allocated/base pointer
aligned/data pointer
offset
size[0]
size[1]
stride[0]
stride[1]
~~~

两个输入按原先的参数顺序连接，共 14 项。这里不是“传入两个 C struct 值”，而是“把两个 descriptor 展成 14 个独立参数”。

来源：

- llvm/mlir/lib/Conversion/LLVMCommon/MemRefBuilder.cpp:247–254：实际 unpack 顺序。
- 同文件 :259–261：数量为 3 + 2 * rank。
- llvm/mlir/lib/Conversion/LLVMCommon/TypeConverter.cpp:784–787、802–812：调用操作数和函数参数展开。
- 同文件 :727–728：单个 memref 返回值转换后保持一个 descriptor struct。

确切的 LLVM IR 类型形式为：

~~~llvm
%M2 = type { ptr, ptr, i64, [2 x i64], [2 x i64] }

; 用声明形式展示类型；若存在 ciface 桥接，实际 @foo 是 private definition。
declare %M2 @foo(
    ptr, ptr, i64, i64, i64, i64, i64,
    ptr, ptr, i64, i64, i64, i64, i64)
~~~

按 C 的字段名字表达其类型含义，可以写成：

~~~c
#include <stdint.h>

typedef struct {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
} M2;

/* C-like 描述：用于核对字段和逻辑参数顺序；
   不保证直接交给任意 C 编译器后就与 RAW LLVM ABI 兼容。 */
M2 foo(
    float *a_allocated, float *a_aligned,
    int64_t a_offset,
    int64_t a_size0, int64_t a_size1,
    int64_t a_stride0, int64_t a_stride1,
    float *b_allocated, float *b_aligned,
    int64_t b_offset,
    int64_t b_size0, int64_t b_size1,
    int64_t b_stride0, int64_t b_stride1);
~~~

offset 和 stride 的单位是元素，不是字节。rank-2 的访问公式为：

~~~c
aligned[offset + i * strides[0] + j * strides[1]]
~~~

allocated 是原始分配地址，主要用于分配/释放语义；访问基址是 aligned。二者可能相同，也可能不同。字段布局和访问公式见 llvm/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h:131–149。

对于“确切 C 签名”，需要明确一条边界：上面的 LLVM signature 是确切的，但不能保证存在一个在所有平台上直接编译就兼容的普通 C aggregate-return 原型。C 前端可能根据目标 ABI 把结构体返回转换成 sret、coercion 等形式，不能只凭 C struct 字段一样就判定二进制兼容。本次没有运行目标 C/LLVM ABI 的机器码交叉检查，因此不对直接链接该 RAW C 原型作保证。

MLIR 自己也因此使用 ciface：TypeConverter.cpp:438–442 明确说明 struct 不能安全地直接经 C interface 返回，于是将它改成首个指针参数。这个首参在 returning-memref 契约下承载的是“返回 descriptor 的存储槽”，不是隐含的已分配 tensor 数据。

建议的验收判据是：输出作为显式 memref operand、函数返回 void；RAW bridge 内每个 descriptor 指针在调用前都有合法初始化；外部 C symbol 是约定的 ciface；kernel 接收到的输出数据地址、offset/stride 和生命周期全部有效。不要再以“模型 body 内有没有 RAW 调用”判断接口是否正确。

