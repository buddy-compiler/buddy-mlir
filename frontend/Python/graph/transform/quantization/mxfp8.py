import torch


def quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(values, torch.Tensor) or values.dtype != torch.float32:
        raise ValueError("MXFP8 requires FP32 Tensor values")
    if values.ndim < 1 or values.shape[-1] == 0 or values.shape[-1] % 32:
        raise ValueError(
            "MXFP8 requires a positive last dimension divisible by 32"
        )
    if not torch.isfinite(values).all():
        raise ValueError("MXFP8 requires finite FP32 values")
    shape = (*values.shape[:-1], values.shape[-1] // 32)
    blocks = values.reshape(*shape, 32)
    maximum = blocks.abs().amax(-1)
    exponent = torch.where(
        maximum == 0, 0, (torch.frexp(maximum)[1] - 9).clamp_min(-127)
    )
    scaled = torch.ldexp(blocks, -exponent[..., None]).contiguous()
    bits = scaled.view(torch.int32)
    sign = (bits >> 24) & 128
    element_exponent = ((bits >> 23) & 255) - 120
    fraction = bits & 0x7FFFFF
    high, low = fraction >> 20, fraction & 0xFFFFF
    rounded = high + ((low > 0x80000) | ((low == 0x80000) & ((high & 1) != 0)))
    codes = torch.where(
        element_exponent <= 0,
        (scaled.abs() * 512).round().to(torch.int32),
        element_exponent * 8 + rounded,
    )
    return (codes.clamp_max(126) | sign).to(torch.int8).reshape(values.shape), (
        exponent + 127
    ).to(torch.int8)


def dequantize(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(codes, torch.Tensor)
        or not isinstance(scales, torch.Tensor)
        or codes.dtype != torch.int8
        or scales.dtype != torch.int8
    ):
        raise ValueError("MXFP8 requires INT8 Tensor codes and scales")
    if codes.ndim < 1 or codes.shape[-1] == 0 or codes.shape[-1] % 32:
        raise ValueError(
            "MXFP8 requires a positive last dimension divisible by 32"
        )
    shape = (*codes.shape[:-1], codes.shape[-1] // 32)
    if scales.shape != shape or scales.device != codes.device:
        raise ValueError("MXFP8 code/scale shape or device mismatch")
    unsigned, scale = codes.to(torch.int16) & 255, scales.to(torch.int16) & 255
    if torch.any((unsigned & 127) == 127) or torch.any(scale == 255):
        raise ValueError("MXFP8 NaN encoding")
    exponent, fraction = (unsigned >> 3) & 15, (unsigned & 7).to(torch.float32)
    elements = torch.where(
        exponent == 0, fraction / 512, torch.ldexp(8 + fraction, exponent - 10)
    )
    elements = torch.where((unsigned & 128) != 0, -elements, elements)
    result = torch.ldexp(
        elements.reshape(*shape, 32), scale[..., None] - 127
    ).reshape(codes.shape)
    if not torch.isfinite(result).all():
        raise ValueError("MXFP8 decoded value overflows FP32")
    return result
