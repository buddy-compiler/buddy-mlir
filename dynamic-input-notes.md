目前已经成功将 prefill 的 input 设置为动态形状，同时修改部分前端算子生成的逻辑以适配该形状。（但需要注意的是，这些修改的正确性目前无法验证）

现在遇到的问题是，在编译`subgraph0_prefill.o`时，会有报错：

```text
[1/6] Building subgraph_prefill-f16.o
/home/cyanic/repos/buddy-mlir/build/examples/BuddyDeepSeekR1/subgraph0_prefill-f16.mlir:136:11: error: 'tosa.mul' op operands don't have matching ranks
    %95 = "tosa.mul"(%arg3, %94) : (tensor<1536xf16>, tensor<1x?x1536xf16>) -> tensor<1x?x1536xf16>
          ^
/home/cyanic/repos/buddy-mlir/build/examples/BuddyDeepSeekR1/subgraph0_prefill-f16.mlir:136:11: note: see current operation: %95 = "tosa.mul"(%arg3, %94) : (tensor<1536xf16>, tensor<1x?x1536xf16>) -> tensor<1x?x1536xf16>
```

可以看到，动态形状已经从 forward_prefill 传导到其他阶段了。仔细看这个报错，可以发现其来自于 `tosa.mul` 的期望参数类型和实际传入参数类型的不一致。而产生这个不一致的原因，我觉得是四个 mlir 文件的*分离生成*，以及相关 tosa 算子没有正确适配动态形状。