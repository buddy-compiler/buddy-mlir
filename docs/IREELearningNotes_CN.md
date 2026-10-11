# IREE的学习记录/通过 IREE 源码探究 Attention 算子的内存优化与增量计算（0）

文件路径：`iree/compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/TileAttention.cpp`。

## 前期准备
### 测试文件生成
文件路径：`iree/tests/e2e/attention`
```
python generate_e2e_attention_tests.py \
    --output_attention_mlir attention.mlir \
    --output_calls_mlir calls.mlir \
    --query_type f16 \
    --key_type f16 \
    --value_type f16 \
    --shapes_scale small
```
生成测试所用脚本：
```
func.func @attention_2_256_64_32_16_dtype_f16_f16_f16_f16(%query: tensor<2x256x64xf16>, %key: tensor<2x32x64xf16>, %value: tensor<2x32x16xf16>, %scale: f32) -> tensor<2x256x16xf16> {
  %result0 = tensor.empty(): tensor<2x256x16xf16>
  %scale_f16 = arith.truncf %scale : f32 to f16
  %result1 = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(batch, m, n, k1, k2) -> (batch, m, k1)>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, k2, k1)>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, k2, n)>,
                       affine_map<(batch, m, n, k1, k2) -> ()>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, m, n)>]
}      ins(%query, %key, %value, %scale_f16: tensor<2x256x64xf16>, tensor<2x32x64xf16>, tensor<2x32x16xf16>, f16)
      outs(%result0: tensor<2x256x16xf16>) {
   ^bb0(%score: f32):
   iree_linalg_ext.yield %score : f32
 } -> tensor<2x256x16xf16>
 return %result1: tensor<2x256x16xf16>
}
```

使用如下命令运行，生成的日志将保存在compile_debug.log中

```Bash
iree-compile attention.mlir \
    --iree-hal-target-backends=llvm-cpu \
    --mlir-print-ir-after-all \
    2>&1 | tee compile_debug.log
```

## 变化分析
这里是`ConvertAttentionToOnlineAttentionPass`变化前：
```MLIR
/ -----// IR Dump After LLVMCPUTileAndFuseProducerConsumerPass (iree-llvmcpu-tile-and-fuse-producer-consumer) //----- //
func.func @attention_2_256_64_32_16_dtype_f16_f16_f16_f16_dispatch_0_attention_2x256x16x64x32() attributes {translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = arith.bitcast %0 : i32 to f32
  %2 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2x256x64xf16, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2x32x64xf16, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2x32x16xf16, #hal.descriptor_type<storage_buffer>>
  %5 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<2x256x16xf16, #hal.descriptor_type<storage_buffer>>
  %6 = iree_codegen.load_from_buffer %2 : memref<2x256x64xf16, #hal.descriptor_type<storage_buffer>> -> tensor<2x256x64xf16>
  %7 = iree_codegen.load_from_buffer %3 : memref<2x32x64xf16, #hal.descriptor_type<storage_buffer>> -> tensor<2x32x64xf16>
  %8 = iree_codegen.load_from_buffer %4 : memref<2x32x16xf16, #hal.descriptor_type<storage_buffer>> -> tensor<2x32x16xf16>
  %9 = tensor.empty() : tensor<2x256x16xf16>
  %10 = arith.truncf %1 : f32 to f16
  %11 = scf.forall (%arg0, %arg1) = (0, 0) to (2, 256) step (1, 32) shared_outs(%arg2 = %9) -> (tensor<2x256x16xf16>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0, %arg1, 0] [1, 32, 16] [1, 1, 1] : tensor<2x256x16xf16> to tensor<1x32x16xf16>
    %12 = scf.forall (%arg3, %arg4) = (0, 0) to (32, 16) step (1, 8) shared_outs(%arg5 = %extracted_slice) -> (tensor<1x32x16xf16>) {
      %13 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg3, %arg1]
      %extracted_slice_0 = tensor.extract_slice %6[%arg0, %13, 0] [1, 1, 64] [1, 1, 1] : tensor<2x256x64xf16> to tensor<1x1x64xf16>
      %extracted_slice_1 = tensor.extract_slice %7[%arg0, 0, 0] [1, 32, 64] [1, 1, 1] : tensor<2x32x64xf16> to tensor<1x32x64xf16>
      %extracted_slice_2 = tensor.extract_slice %8[%arg0, 0, %arg4] [1, 32, 8] [1, 1, 1] : tensor<2x32x16xf16> to tensor<1x32x8xf16>
      %extracted_slice_3 = tensor.extract_slice %arg5[0, %arg3, %arg4] [1, 1, 8] [1, 1, 1] : tensor<1x32x16xf16> to tensor<1x1x8xf16>
      %14 = iree_linalg_ext.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>], lowering_config = #iree_cpu.lowering_config<distribution = [1, 32, 16, 0, 0], vector_common_parallel = [1, 1, 8, 0, 0], vector_reduction = [0, 0, 0, 0, 2]>} ins(%extracted_slice_0, %extracted_slice_1, %extracted_slice_2, %10 : tensor<1x1x64xf16>, tensor<1x32x64xf16>, tensor<1x32x8xf16>, f16) outs(%extracted_slice_3 : tensor<1x1x8xf16>) {
      ^bb0(%arg6: f32):
        iree_linalg_ext.yield %arg6 : f32
      } -> tensor<1x1x8xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg5[%c0, %arg3, %arg4] [1, 1, 8] [1, 1, 1] : tensor<1x1x8xf16> into tensor<1x32x16xf16>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %12 into %arg2[%arg0, %arg1, 0] [1, 32, 16] [1, 1, 1] : tensor<1x32x16xf16> into tensor<2x256x16xf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %11, %5 : tensor<2x256x16xf16> into memref<2x256x16xf16, #hal.descriptor_type<storage_buffer>>
  return
}
```

这里是`ConvertAttentionToOnlineAttentionPass`变换后：
```MLIR
// -----// IR Dump After ConvertAttentionToOnlineAttentionPass (iree-linalg-ext-convert-attention-to-online-attention) //----- //
func.func @attention_2_256_64_32_16_dtype_f16_f16_f16_f16_dispatch_0_attention_2x256x16x64x32() attributes {translation_info = #iree_codegen.translation_info<pipeline = CPULinalgExtTileAndVectorize>} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
  %1 = arith.bitcast %0 : i32 to f32
  %2 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2x256x64xf16, #hal.descriptor_type<storage_buffer>>
  %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2x32x64xf16, #hal.descriptor_type<storage_buffer>>
  %4 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2x32x16xf16, #hal.descriptor_type<storage_buffer>>
  %5 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(3) alignment(64) offset(%c0) flags(Indirect) : memref<2x256x16xf16, #hal.descriptor_type<storage_buffer>>
  %6 = iree_codegen.load_from_buffer %2 : memref<2x256x64xf16, #hal.descriptor_type<storage_buffer>> -> tensor<2x256x64xf16>
  %7 = iree_codegen.load_from_buffer %3 : memref<2x32x64xf16, #hal.descriptor_type<storage_buffer>> -> tensor<2x32x64xf16>
  %8 = iree_codegen.load_from_buffer %4 : memref<2x32x16xf16, #hal.descriptor_type<storage_buffer>> -> tensor<2x32x16xf16>
  %9 = tensor.empty() : tensor<2x256x16xf16>
  %10 = arith.truncf %1 : f32 to f16
  %11 = scf.forall (%arg0, %arg1) = (0, 0) to (2, 256) step (1, 32) shared_outs(%arg2 = %9) -> (tensor<2x256x16xf16>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg0, %arg1, 0] [1, 32, 16] [1, 1, 1] : tensor<2x256x16xf16> to tensor<1x32x16xf16>
    %12 = scf.forall (%arg3, %arg4) = (0, 0) to (32, 16) step (1, 8) shared_outs(%arg5 = %extracted_slice) -> (tensor<1x32x16xf16>) {
      %13 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%arg3, %arg1]
      %extracted_slice_0 = tensor.extract_slice %6[%arg0, %13, 0] [1, 1, 64] [1, 1, 1] : tensor<2x256x64xf16> to tensor<1x1x64xf16>
      %extracted_slice_1 = tensor.extract_slice %7[%arg0, 0, 0] [1, 32, 64] [1, 1, 1] : tensor<2x32x64xf16> to tensor<1x32x64xf16>
      %extracted_slice_2 = tensor.extract_slice %8[%arg0, 0, %arg4] [1, 32, 8] [1, 1, 1] : tensor<2x32x16xf16> to tensor<1x32x8xf16>
      %extracted_slice_3 = tensor.extract_slice %arg5[0, %arg3, %arg4] [1, 1, 8] [1, 1, 1] : tensor<1x32x16xf16> to tensor<1x1x8xf16>
      %14 = tensor.empty() : tensor<1x1x8xf32>
      %15 = tensor.empty() : tensor<1x1xf32>
      %cst = arith.constant 0.000000e+00 : f32
      %cst_4 = arith.constant -3.40282347E+38 : f32
      %cst_5 = arith.constant 0.000000e+00 : f32
      %16 = linalg.fill ins(%cst : f32) outs(%14 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
      %17 = linalg.fill ins(%cst_4 : f32) outs(%15 : tensor<1x1xf32>) -> tensor<1x1xf32>
      %18 = linalg.fill ins(%cst_5 : f32) outs(%15 : tensor<1x1xf32>) -> tensor<1x1xf32>
      %19:3 = iree_linalg_ext.online_attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>], lowering_config = #iree_cpu.lowering_config<distribution = [1, 32, 16, 0, 0], vector_common_parallel = [1, 1, 8, 0, 0], vector_reduction = [0, 0, 0, 0, 2]>} ins(%extracted_slice_0, %extracted_slice_1, %extracted_slice_2, %10 : tensor<1x1x64xf16>, tensor<1x32x64xf16>, tensor<1x32x8xf16>, f16) outs(%16, %17, %18 : tensor<1x1x8xf32>, tensor<1x1xf32>, tensor<1x1xf32>) {
      ^bb0(%arg6: f32):
        iree_linalg_ext.yield %arg6 : f32
      } -> tensor<1x1x8xf32>, tensor<1x1xf32>, tensor<1x1xf32>
      %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19#2, %19#0 : tensor<1x1xf32>, tensor<1x1x8xf32>) outs(%extracted_slice_3 : tensor<1x1x8xf16>) {
      ^bb0(%in: f32, %in_6: f32, %out: f16):
        %cst_7 = arith.constant 1.000000e+00 : f32
        %21 = arith.divf %cst_7, %in : f32
        %22 = arith.mulf %21, %in_6 : f32
        %23 = arith.truncf %22 : f32 to f16
        linalg.yield %23 : f16
      } -> tensor<1x1x8xf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %20 into %arg5[%c0, %arg3, %arg4] [1, 1, 8] [1, 1, 1] : tensor<1x1x8xf16> into tensor<1x32x16xf16>
      }
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %12 into %arg2[%arg0, %arg1, 0] [1, 32, 16] [1, 1, 1] : tensor<1x32x16xf16> into tensor<2x256x16xf16>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
  iree_codegen.store_to_buffer %11, %5 : tensor<2x256x16xf16> into memref<2x256x16xf16, #hal.descriptor_type<storage_buffer>>
  return
}
```
![](./Images/Exploring_memory_optimization_and_incremental_computation_of_the_attention_operator_through_the_IREE_source_code.png)

### 转换前的 Attention 操作
```MLIR
%14 = iree_linalg_ext.attention {indexing_maps = [...], lowering_config = ...}
      ins(%extracted_slice_0, %extracted_slice_1, %extracted_slice_2, %10
          : tensor<1x1x64xf16>, tensor<1x32x64xf16>, tensor<1x32x8xf16>, f16)
      outs(%extracted_slice_3 : tensor<1x1x8xf16>)
```
特点：
- 这是一个整体操作，一次处理整个 Key 序列（长度 32）。
- 隐含了标准 Attention 的三步：Q·K^T → softmax → 加权 V。
- 输出直接是归一化后的结果（f16）。
- 无法在内部对 Key 序列进行分块，因为 softmax 需要全局统计量（最大值、和）。

局限性：

- 需要一次加载所有 Key 和 Value，内存占用大（O(seq_len_k)）。
- 如果 Key 序列很长，会导致缓存压力大，难以利用数据局部性。
- 难以进一步向量化和并行化，因为 softmax 依赖于全局归约。

### 转换后的 Online Attention
转换后，上述操作被拆分为三部分：

#### 1. 初始化累加器（TileAttention.cpp:63-96）

```cpp
// 源码位置：TileAttention.cpp:74-96
Type f32Type = rewriter.getF32Type();
Value acc = tensor::EmptyOp::create(rewriter, loc, accSize, f32Type);
Value rowRedEmpty = tensor::EmptyOp::create(rewriter, loc, rowRedSize, f32Type);

Value accInit =
    arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type, rewriter,
                            loc, /*useOnlyFiniteValue=*/true);  // 0.0
Value maxInit =
    arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type, rewriter,
                            loc, /*useOnlyFiniteValue=*/true);  // -inf
Value sumInit = arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type,
                                        rewriter, loc);  // 0.0

Value accFill = linalg::FillOp::create(rewriter, loc, ValueRange{accInit}, acc).getResult(0);
Value maxFill = linalg::FillOp::create(rewriter, loc, ValueRange{maxInit}, rowRedEmpty).getResult(0);
Value sumFill = linalg::FillOp::create(rewriter, loc, ValueRange{sumInit}, rowRedEmpty).getResult(0);
```

**关键设计决策**：

1. **使用原子操作的 Identity 值**：
   - `arith::getIdentityValue(arith::AtomicRMWKind::addf)` → 返回加法恒等值 `0.0`
   - `arith::getIdentityValue(arith::AtomicRMWKind::maximumf)` → 返回最大值恒等值 `-inf`
   - 这保证了数值稳定性和可组合性

2. **维度计算逻辑**：
   ```cpp
   // TileAttention.cpp:49-72
   AffineMap maxMap = AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, ctx);
   for (auto dim : llvm::concat<const int64_t>(opInfo.getBatchDims(), opInfo.getMDims())) {
     maxMap = maxMap.insertResult(rewriter.getAffineDimExpr(dim), maxMap.getNumResults());
   }
   ```
   - `maxMap` 和 `sumMap` 只包含 `(batch, m)` 维度
   - `accMap` 使用原始的输出映射 `(batch, m, n)`

3. **累加器的作用**：
   - `%16`：累加器（acc），初始化为 0，用于累积加权后的 Value（形状 `1x1x8`）
   - `%17`：最大值累加器（m），初始化为 `-INF`（-3.4e38），用于记录当前已处理 Key 块中的最大 QK 值
   - `%18`：指数和累加器（s），初始化为 0，用于记录当前已处理 Key 块的指数和（`sum(exp(QK - m))`）
2. Online Attention 核心计算（TileAttention.cpp:98-116）

```cpp
// 源码位置：TileAttention.cpp:98-116
// Create online attention op.
SmallVector<AffineMap> indexingMaps = attnOp.getIndexingMapsArray();
indexingMaps.push_back(maxMap);  // 添加 max 的映射
indexingMaps.push_back(sumMap);  // 添加 sum 的映射

Value mask = attnOp.getMask() ? attnOp.getMask() : Value();

OnlineAttentionOp onlineAttn = OnlineAttentionOp::create(
    rewriter, loc,
    TypeRange{accFill.getType(), maxFill.getType(), sumFill.getType()},
    attnOp.getQuery(), attnOp.getKey(), attnOp.getValue(), attnOp.getScale(),
    mask, accFill, maxFill, sumFill,
    rewriter.getAffineMapArrayAttr(indexingMaps),
    attnOp.getDecompositionConfigAttr());

// 复制原始 attention op 的计算逻辑
rewriter.cloneRegionBefore(attnOp.getRegion(), onlineAttn.getRegion(),
                           onlineAttn.getRegion().begin());
onlineAttn->setDiscardableAttrs(attnOp->getDiscardableAttrDictionary());
```

**MLIR 代码**：
```MLIR
%19:3 = iree_linalg_ext.online_attention {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,  // Q
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,  // K
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2)>,  // V
      affine_map<(d0, d1, d2, d3, d4) -> ()>,            // scale
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,  // output
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>,      // max (新增)
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>       // sum (新增)
    ],
    lowering_config = #iree_cpu.lowering_config<...>
}
ins(%extracted_slice_0, %extracted_slice_1, %extracted_slice_2, %10
    : tensor<1x1x64xf16>, tensor<1x32x64xf16>, tensor<1x32x8xf16>, f16)
outs(%16, %17, %18
    : tensor<1x1x8xf32>, tensor<1x1xf32>, tensor<1x1xf32>)
-> tensor<1x1x8xf32>, tensor<1x1xf32>, tensor<1x1xf32>
```

**关键实现细节**：

1. **Indexing Maps 扩展**：
   ```cpp
   indexingMaps.push_back(maxMap);  // 第6个映射：max
   indexingMaps.push_back(sumMap);  // 第7个映射：sum
   ```
   - 原始 attention 有 5 个映射（Q, K, V, scale, output）
   - 在线 attention 扩展为 7 个映射（+ max, + sum）
   - 这使得 online attention 可以访问和更新额外的状态

2. **Mask 处理**：
   ```cpp
   Value mask = attnOp.getMask() ? attnOp.getMask() : Value();
   ```
   - 如果原始 attention 有 mask，会传递给 online attention
   - 在 online softmax 算法中，mask 会被加到 QK 结果上

3. **Region 克隆**：
   ```cpp
   rewriter.cloneRegionBefore(attnOp.getRegion(), onlineAttn.getRegion(),
                          onlineAttn.getRegion().begin());
   ```
   - 复制原始 attention 的计算逻辑到 online attention
   - 这个 region 包含对 score 的处理（如 mask 应用）

4. **属性传递**：
   ```cpp
   onlineAttn->setDiscardableAttrs(attnOp->getDiscardableAttrDictionary());
   ```
   - 保留原始 attention 的所有属性
   - 包括 lowering_config 等优化信息

**输出含义**：
- `%19#0`：更新后的累加器（acc'），形状 `1x1x8xf32`
- `%19#1`：更新后的最大值（m'），形状 `1x1xf32`
- `%19#2`：更新后的和（s'），形状 `1x1xf32`
3. 后处理归一化（TileAttention.cpp:118-148）

```cpp
// 源码位置：TileAttention.cpp:118-148
Value x = onlineAttn.getResult(0);      // acc'
Value sum = onlineAttn.getResult(2);    // s'

// Merge the outputs of online attention:
//  x = (1 / sum) * x

// Compress the indexing maps.
SmallVector<AffineMap> compressedMaps =
    compressUnusedDims(SmallVector<AffineMap>{sumMap, accMap, accMap});

SmallVector<utils::IteratorType> iteratorTypes(compressedMaps[0].getNumDims(),
                                               utils::IteratorType::parallel);

auto genericOp = linalg::GenericOp::create(
    rewriter, loc, attnOp.getOutput().getType(), ValueRange{sum, x},
    attnOp.getOutput(), compressedMaps, iteratorTypes,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value one = arith::ConstantOp::create(
          b, loc, b.getFloatAttr(args[0].getType(), 1.0));
      Value reciprocal = arith::DivFOp::create(b, loc, one, args[0]);
      // Both sum and x are in fp32, as created earlier, so we only need
      // to cast after the mul.
      Value result = arith::MulFOp::create(b, loc, reciprocal, args[1]);
      // Cast result to the required type by attention output.
      result = convertScalarToDtype(b, loc, result, args[2].getType(),
                                    /*isUnsignedCast=*/false);
      linalg::YieldOp::create(b, loc, result);
    });

rewriter.replaceOp(attnOp, genericOp);
```

**MLIR 代码**：
```MLIR
%20 = linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d1)>,      // sum 映射
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>,   // acc 映射
    affine_map<(d0, d1, d2) -> (d0, d1, d2)>    // output 映射
  ],
  iterator_types = ["parallel", "parallel", "parallel"]
} ins(%19#2, %19#0 : tensor<1x1xf32>, tensor<1x1x8xf32>)
   outs(%extracted_slice_3 : tensor<1x1x8xf16>) {
  ^bb0(%in: f32, %in_6: f32, %out: f16):
    %cst_7 = arith.constant 1.000000e+00 : f32
    %21 = arith.divf %cst_7, %in : f32          // 1/sum
    %22 = arith.mulf %21, %in_6 : f32           // (1/sum) * acc
    %23 = arith.truncf %22 : f32 to f16         // 转回 f16
    linalg.yield %23 : f16
} -> tensor<1x1x8xf16>
```

**归一化算法详解**：

1. **数学原理**：
   ```
   最终输出 = acc / sum
   其中：
   - acc = Σ (exp(QK_i - m_i) * V_i)
   - sum = Σ exp(QK_i - m_i)
   - m_i 是全局最大值（通过 online softmax 维护）
   ```

2. **Indexing Maps 压缩**：
   ```cpp
   compressUnusedDims(SmallVector<AffineMap>{sumMap, accMap, accMap});
   ```
   - 移除未使用的维度，优化访问模式
   - `sumMap` 和 `accMap` 都被压缩到 `(batch, m)` 和 `(batch, m, n)`

3. **类型转换**：
   ```cpp
   result = convertScalarToDtype(b, loc, result, args[2].getType(),
                                 /*isUnsignedCast=*/false);
   ```
   - 从 f32 转换回原始类型（f16）
   - 使用 `isUnsignedCast=false` 表示有符号浮点数转换

4. **并行化设计**：
   ```cpp
   SmallVector<utils::IteratorType> iteratorTypes(compressedMaps[0].getNumDims(),
                                                  utils::IteratorType::parallel);
   ```
   - 所有迭代器类型都是 `parallel`
   - 这使得归一化操作可以完全并行化

## 转换前后的对比

### Pass 执行流程分析（TileAttention.cpp:151-158）

```cpp
void ConvertAttentionToOnlineAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  getOperation()->walk([&](AttentionOp attnOp) {
    SmallVector<Operation *> ops;
    convertToOnlineAttention(attnOp, ops, rewriter);
  });
}
```

**执行流程**：

1. **遍历所有 Attention 操作**：
   - 使用 `getOperation()->walk()` 遍历整个函数
   - 查找所有 `AttentionOp` 类型的操作

2. **逐个转换**：
   - 对每个找到的 attention op 调用 `convertToOnlineAttention`
   - 收集生成的操作到 `ops` 向量中

3. **IR 重写**：
   - 使用 `IRRewriter` 进行安全的 IR 变换
   - 确保 SSA 形式和类型一致性

### 核心转换函数详解（TileAttention.cpp:35-149）

#### 阶段 1：分析 Attention 操作（35-47 行）

```cpp
FailureOr<AttentionOpDetail> maybeOpInfo =
    AttentionOpDetail::get(attnOp.getQueryMap(), attnOp.getKeyMap(),
                           attnOp.getValueMap(), attnOp.getOutputMap());
assert(succeeded(maybeOpInfo) && "Invalid attention indexing maps");
AttentionOpDetail opInfo = maybeOpInfo.value();
```

**维度推断逻辑**（IndexingUtils.cpp:29-80）：

```cpp
// Q   = B x M x K1
// K   = B x K2 x K1
// V   = B x K2 x N
// O   = B x M x N

// 通过集合运算推断各维度
// B = (Q & V) U (K & O)
// K1 = Q & K - B
// K2 = K - B - K1
// M = Q - B - K1
// N = V - B - K2
```

**关键设计**：
- 使用集合运算而不是硬编码维度索引
- 支持任意的维度排列（transpose）
- 验证 indexing maps 的有效性

#### 阶段 2：创建 Indexing Maps（49-72 行）

```cpp
// 创建 max 和 sum 的映射：(batch, m)
AffineMap maxMap = AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, ctx);
for (auto dim :
     llvm::concat<const int64_t>(opInfo.getBatchDims(), opInfo.getMDims())) {
  maxMap = maxMap.insertResult(rewriter.getAffineDimExpr(dim),
                               maxMap.getNumResults());
}
AffineMap sumMap = maxMap;
```

**映射构建过程**：
1. 创建空的 affine map
2. 遍历 batch 和 m 维度的索引
3. 将这些维度插入到 map 的结果中
4. maxMap 和 sumMap 共享相同的结构

#### 阶段 3：计算张量大小（67-72 行）

```cpp
SmallVector<OpFoldResult> sizes =
    llvm::map_to_vector(domain, [](Range x) { return x.size; });
SmallVector<OpFoldResult> accSize =
    applyPermutationMap<OpFoldResult>(accMap, sizes);
SmallVector<OpFoldResult> rowRedSize =
    applyPermutationMap<OpFoldResult>(maxMap, sizes);
```

**大小计算**：
- `domain`：迭代域的完整大小
- `accSize`：通过 accMap 重排得到的大小 `(batch, m, n)`
- `rowRedSize`：通过 maxMap 重排得到的大小 `(batch, m)`

#### 阶段 4：创建并填充累加器（74-96 行）

```cpp
// 使用原子操作的 identity 值
Value accInit =
    arith::getIdentityValue(arith::AtomicRMWKind::addf, f32Type, rewriter,
                            loc, /*useOnlyFiniteValue=*/true);  // 0.0
Value maxInit =
    arith::getIdentityValue(arith::AtomicRMWKind::maximumf, f32Type, rewriter,
                            loc, /*useOnlyFiniteValue=*/true);  // -inf
```

**为什么使用 Identity 值**：
- **数值稳定性**：避免硬编码的魔法数字
- **语义正确性**：identity 值是归约操作的初始值
- **可组合性**：支持后续的分块和并行化

#### 阶段 5：创建 Online Attention 操作（98-116 行）

```cpp
SmallVector<AffineMap> indexingMaps = attnOp.getIndexingMapsArray();
indexingMaps.push_back(maxMap);
indexingMaps.push_back(sumMap);

OnlineAttentionOp onlineAttn = OnlineAttentionOp::create(
    rewriter, loc,
    TypeRange{accFill.getType(), maxFill.getType(), sumFill.getType()},
    attnOp.getQuery(), attnOp.getKey(), attnOp.getValue(), attnOp.getScale(),
    mask, accFill, maxFill, sumFill,
    rewriter.getAffineMapArrayAttr(indexingMaps),
    attnOp.getDecompositionConfigAttr());
```

**关键设计**：
- **类型一致性**：所有输出都是 f32 类型
- **属性传递**：保留 decomposition_config
- **Region 复制**：保持原始计算逻辑

#### 阶段 6：创建归一化操作（118-148 行）

```cpp
auto genericOp = linalg::GenericOp::create(
    rewriter, loc, attnOp.getOutput().getType(), ValueRange{sum, x},
    attnOp.getOutput(), compressedMaps, iteratorTypes,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value one = arith::ConstantOp::create(
          b, loc, b.getFloatAttr(args[0].getType(), 1.0));
      Value reciprocal = arith::DivFOp::create(b, loc, one, args[0]);
      Value result = arith::MulFOp::create(b, loc, reciprocal, args[1]);
      result = convertScalarToDtype(b, loc, result, args[2].getType(),
                                    /*isUnsignedCast=*/false);
      linalg::YieldOp::create(b, loc, result);
    });
```

**优化技巧**：
1. 使用 `1 / sum` 而不是 `sum⁻¹`（避免幂运算）
2. 类型转换在乘法之后进行（提高精度）
3. 使用 lambda 表达式简化代码

#### 阶段 7：替换原始操作（148 行）

```cpp
rewriter.replaceOp(attnOp, genericOp);
```

**IR 重写机制**：
- 使用 `rewriter.replaceOp` 替换原始操作
- 自动维护 SSA 形式
- 确保类型和 use-def 链的正确性

### 维度推断机制详解

IREE 使用巧妙的集合运算来推断 attention 操作的维度，而不是硬编码维度索引。这使得实现能够支持任意的维度排列。

#### 集合运算推断逻辑（IndexingUtils.cpp:29-80）

```cpp
void AttentionOpDetail::inferFromIndexingMaps(AffineMap qMap, AffineMap kMap,
                                              AffineMap vMap, Affine oMap) {
  // 提取每个操作使用的维度集合
  llvm::SmallDenseSet<int64_t> qSet = findPermutationsIndexingOperand(qMap);
  llvm::SmallDenseSet<int64_t> kSet = findPermutationsIndexingOperand(kMap);
  llvm::SmallDenseSet<int64_t> vSet = findPermutationsIndexingOperand(vMap);
  llvm::SmallDenseSet<int64_t> oSet = findPermutationsIndexingOperand(oMap);

  // Batch 维度：同时出现在 Q&V 和 K&O 中
  llvm::SmallDenseSet<int64_t> b1Set = qSet;
  llvm::set_intersect(b1Set, vSet);
  llvm::SmallDenseSet<int64_t> b2Set = kSet;
  llvm::set_intersect(b2Set, oSet);
  llvm::SmallDenseSet<int64_t> bSet = b1Set;
  llvm::set_union(bSet, b2Set);

  // K1 维度：Q 和 K 共有，但不是 batch
  llvm::SmallDenseSet<int64_t> k1Set = qSet;
  llvm::set_intersect(k1Set, kSet);
  llvm::set_subtract(k1Set, bSet);

  // K2 维度：K 中剩余的维度
  llvm::SmallDenseSet<int64_t> k2Set = kSet;
  llvm::set_subtract(k2Set, bSet);
  llvm::set_subtract(k2Set, k1Set);

  // M 维度：Q 中剩余的维度
  llvm::SmallDenseSet<int64_t> mSet = qSet;
  llvm::set_subtract(mSet, bSet);
  llvm::set_subtract(mSet, k1Set);

  // N 维度：V 中剩余的维度
  llvm::SmallDenseSet<int64_t> nSet = vSet;
  llvm::set_subtract(nSet, bSet);
  llvm::set_subtract(nSet, k2Set);

  // 排序以确保维度从外到内
  llvm::sort(batch);
  llvm::sort(m);
  llvm::sort(k1);
  llvm::sort(k2);
  llvm::sort(n);
}
```

**推断规则总结**：

| 维度 | 推断规则 | 说明 |
|------|---------|------|
| Batch (B) | `(Q ∧ V) ∪ (K ∧ O)` | 同时在 Q,V 和 K,O 中出现的维度 |
| M | `Q - B - K1` | Q 中除去 B 和 K1 的维度 |
| K1 | `Q ∧ K - B` | Q 和 K 共有但不是 B 的维度 |
| K2 | `K - B - K1` | K 中剩余的维度 |
| N | `V - B - K2` | V 中除去 B 和 K2 的维度 |

**示例分析**：

对于测试用例：
```
Q: tensor<2x256x64xf16>        // batch=2, m=256, k1=64
K: tensor<2x32x64xf16>         // batch=2, k2=32, k1=64
V: tensor<2x32x16xf16>         // batch=2, k2=32, n=16
O: tensor<2x256x16xf16>        // batch=2, m=256, n=16
```

维度映射：
```
Q: (d0, d1, d2) → (d0, d1, d2)
K: (d0, d3, d2) → (d0, d3, d2)
V: (d0, d3, d4) → (d0, d3, d4)
O: (d0, d1, d4) → (d0, d1, d4)
```

集合运算：
- `qSet = {0, 1, 2}`（Q 使用的索引）
- `kSet = {0, 3, 2}`（K 使用的索引）
- `vSet = {0, 3, 4}`（V 使用的索引）
- `oSet = {0, 1, 4}`（O 使用的索引）

推断结果：
- `bSet = {0}` → batch 维度：`d0`
- `k1Set = {2}` → k1 维度：`d2`（64）
- `k2Set = {3}` → k2 维度：`d3`（32）
- `mSet = {1}` → m 维度：`d1`（256）
- `nSet = {4}` → n 维度：`d4`（16）

**Indexing Maps 示例**：

```mlir
affine_map<(batch, m, n, k1, k2) -> (batch, m, k1)>   // Q 映射
affine_map<(batch, m, n, k1, k2) -> (batch, k2, k1)>   // K 映射
affine_map<(batch, m, n, k1, k2) -> (batch, k2, n)>    // V 映射
affine_map<(batch, m, n, k1, k2) -> ()>               // scale 映射
affine_map<(batch, m, n, k1, k2) -> (batch, m, n)>    // output 映射
```

**Online Attention 的扩展映射**：

```mlir
affine_map<(batch, m, n, k1, k2) -> (batch, m)>       // max 映射（新增）
affine_map<(batch, m, n, k1, k2) -> (batch, m)>       // sum 映射（新增）
```

这些额外的映射用于：
- `maxMap`：跟踪每个 (batch, m) 位置的最大 QK 值
- `sumMap`：跟踪每个 (batch, m) 位置的 exp(QK - max) 之和
| 方面 | 转换前（单一 Attention） | 转换后（Online Attention） |
| --- | --- | --- |
| 计算粒度 | 一次性处理整个 Key 序列 | 可逐步累积，支持分块处理 |
| 内存需求 | 需要存储完整的注意力矩阵（QK^T） | 只需保持三个小型累加器（与输出块同尺寸） |
| 数值稳定性 | 标准 softmax 使用全局最大值 | 在线 softmax 通过更新 m 和 s 保证稳定 |
| 输出类型 | 直接输出 f16 | 累加器为 f32（更高精度），最后转 f16 |
| 可扩展性 | 难以处理长序列 | 可通过循环分块处理任意长 Key 序列 |
| 后续优化 | 难以进一步 tile | 可对 Key 维进行 tile，并在循环中复用累加器 |


## 总结

IREE 的 `ConvertAttentionToOnlineAttentionPass` 将标准的 Attention 操作转换为在线 softmax 形式，通过精巧的 IR 变换实现了内存优化的 attention 计算。这个 Pass 的核心价值在于：

### 关键技术优势

1. **内存效率**：
   - 从 O(seq_len²) 的内存复杂度降低到 O(seq_len)
   - 不再需要存储完整的 attention 矩阵
   - 只需维护三个小型累加器（acc, max, sum）

2. **可扩展性**：
   - 通过显式的初始化、增量更新和最终归一化
   - 为后续对 K2 维度进行分块扫清了障碍
   - 支持长序列的高效处理

3. **数值稳定性**：
   - 使用 `arith::getIdentityValue` 获取正确的初始值
   - 在线 softmax 算法通过动态更新最大值避免溢出
   - 累加器使用 f32 精度减少累积误差

4. **向量化友好**：
   - 分解后的操作都具有规则的并行模式
   - 所有迭代器类型都是 `parallel`
   - 便于后端生成 SIMD 指令

### 实现亮点

1. **智能维度推断**：
   - 使用集合运算而非硬编码维度索引
   - 支持任意的维度排列（transpose）
   - 自动验证 indexing maps 的有效性

2. **类型安全**：
   - 统一使用 f32 进行计算
   - 在最后一步转换回原始类型
   - 避免硬编码的魔法数字

3. **IR 重写机制**：
   - 使用 `IRRewriter` 进行安全的 IR 变换
   - 自动维护 SSA 形式
   - 正确处理 use-def 链

4. **属性传递**：
   - 保留原始 attention 的所有属性
   - 包括 `lowering_config` 和 `decomposition_config`
   - 支持后续的优化 pass

### 与 Flash Attention V2 的对应

IREE 的实现完全遵循 Flash Attention V2 论文（arXiv:2307.08691）：
- ✅ 支持 K2 维度的分块
- ✅ 使用在线 softmax 算法
- ✅ 保持数值精度
- ✅ 优化内存访问模式

这个 Pass 为 IREE 处理长序列 attention 计算（如 16K、32K 长度）奠定了基础，特别是在内存受限的环境中。
