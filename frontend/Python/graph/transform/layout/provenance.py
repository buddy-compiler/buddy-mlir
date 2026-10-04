from buddy_mlir import ir


def has_nhwc_layout(value):
    """Follow operations that preserve TOSA convolution/pooling axis order."""
    if not isinstance(value, ir.OpResult):
        return False
    producer = value.owner
    if isinstance(producer, ir.OpView):
        producer = producer.operation
    if producer.name in {
        "tosa.conv2d",
        "tosa.depthwise_conv2d",
        "tosa.transpose_conv2d",
        "tosa.avg_pool2d",
        "tosa.max_pool2d",
    }:
        return True
    if producer.name in {
        "tosa.add",
        "tosa.sub",
        "tosa.mul",
        "tosa.maximum",
        "tosa.minimum",
        "tosa.clamp",
        "tosa.sigmoid",
        "tosa.tanh",
        "tosa.exp",
        "tosa.log",
        "tosa.negate",
        "tosa.abs",
        "tosa.reciprocal",
        "tosa.rsqrt",
        "tosa.cast",
        "tosa.select",
        "tosa.slice",
        "tosa.pad",
        "tensor.cast",
        "buddy_trace.end",
    }:
        return any(
            isinstance(operand.type, ir.RankedTensorType)
            and len(ir.RankedTensorType(operand.type).shape) == 4
            and has_nhwc_layout(operand)
            for operand in producer.operands
        )
    return False
