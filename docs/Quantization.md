# Large Model Quantization
## Brief Summary

This proposal extends the quantization bit of large models in Multi-Level Intermediate Representation.

In previous work, we implemented the conversion of large models from pytorch ATen ir level to mlir.

The main focus of our proposal is to implement support for large model quantization at the ATen IR level and at the MLIR level.

## Motivation/Context

On the one hand,large models based on transformers have shown excellent performance on various benchmarks. However, the large model size leads to the high serving costs. 

On the other hand, the size of large models makes it difficult to run the models on small machines that users often use.

Currently,the model size limits the applicable environment of large models, and the model inference speed limits the performance of large models.

Besides,certain specific hardware often has efficient acceleration capabilities. So we can use this capability to achieve acceleration effects on large models.

In this case, we propose quantitative support for this proposal.

## Proposal

### Quantization Type

#### BF16/F16

We plan to implement support for such quantified data types on the CPU.

Taking the PyTorch 2.X importer as an example, AI models can be imported by mapping ATen IR and adapting the TorchDynamo interface.

We can implement support for BF16 and F16 in the conversion process of FX graph to mlir.

For example,

```
def matmul_op(
    node: torch.fx.Node,
    symbol_table: Dict[Tuple[str, int], ir.Operation],
):
    """
    Import the tensor matmul operation.
    From PyTorch `aten.mm.default` operator to MLIR linalg `matmul` operation.

    Note: This op, compute input node's matrix multiplication result.
    Args:
        node: Containing information from the input graph node.
        symbol_table: A dictionary mapping symbols to their corresponding
        operations.

    Returns:
        op: The operation return the linalg.matmul op.
    """
    assert len(node.args) == 2
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    if input1 is None or input2 is None:
        return

    output_shape = list(node.meta["tensor_meta"].shape)
    dtype = str(node.meta["tensor_meta"].dtype)
    if dtype == "torch.float32":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.F32Type.get())
        f32 = ir.F32Type.get()
        element = ir.FloatAttr.get(f32, 0.0)
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        matmul_result_buffer = arith.ConstantOp(tensor_type, attr).result
        op = linalg.matmul(input1, input2, outs=[matmul_result_buffer])
    elif dtype == "torch.bfloat16":
        tensor_type = ir.RankedTensorType.get(output_shape, ir.BF16Type.get())
        bf16 = ir.BF16Type.get()
        element = ir.FloatAttr.get(bf16, 0.0)
        attr = ir.DenseElementsAttr.get_splat(tensor_type, element)
        matmul_result_buffer = arith.ConstantOp(tensor_type, attr).result
        op = linalg.matmul(input1, input2, outs=[matmul_result_buffer])
    return op
```

#### INT

We plan to implement support for such quantified data types on the GPU.

In the first method, we can implement it in Pytorch or use a quantization method, transfer it to FX Graph, and then align it to the MLIR level through subsequent conversion for subsequent execution.

In the second method, we can implement the quantization method at the MLIR level and transfer it to the GPU for execution to achieve MLIR-side GPU quantization inference.

### Optimization

#### Quantization Method

We can implement different quantization methods during the conversion process to achieve different degrees of quantization compression and acceleration effects, such as outlier quantization.

There is a small fraction of salient weights,but skipping the quantization of these salient weights will significantly reduce the quantization loss.

We focus on the powerful role that outlier quantification plays in the quantification process.

#### Quantization Calculation

Mixed-precision calculations appear during the quantization process. 

For example, the processing of outlier quantization introduces a part of mixed-precision calculations, and we are also concerned about this part. 

How to transfer mixed-precision calculations to mlir for processing and convert them into special operator calculations is also one of our topics.

In addition, the calculation optimization of conventional quantification is also one of our topics.

### Test Cases

In order to verify the correctness of each operator we implement, we need to add a series of test cases. 

First, we need to verify that we can generate the mlir code correctly. 

Second, we need to verify that the mlir code we generated can be executed correctly. 

Third, we need to verify that our end-to-end model program can output a correct result.