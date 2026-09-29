import numpy as np


def quantize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] % 32 or not np.isfinite(values).all():
        raise ValueError(
            "MXFP8 requires finite values and a last dimension divisible by 32"
        )
    blocks = values.reshape(*values.shape[:-1], -1, 32)
    maximum = np.max(np.abs(blocks), axis=-1)
    exponent = np.where(
        maximum == 0, 0, np.maximum(np.frexp(maximum)[1] - 9, -127)
    )
    scaled = np.ldexp(blocks, -exponent[..., None]).astype(np.float32)
    bits = scaled.view(np.uint32)
    sign = ((bits >> 24) & 128).astype(np.uint8)
    element_exponent = ((bits >> 23) & 255).astype(np.int32) - 120
    fraction = bits & 0x7FFFFF
    high, low = fraction >> 20, fraction & 0xFFFFF
    rounded = high + ((low > 0x80000) | ((low == 0x80000) & ((high & 1) != 0)))
    codes = element_exponent * 8 + rounded.astype(np.int32)
    subnormal = element_exponent <= 0
    codes[subnormal] = np.rint(
        np.abs(scaled[subnormal]) * np.float32(512)
    ).astype(np.int32)
    codes = np.minimum(codes, 126).astype(np.uint8) | sign
    return codes.reshape(values.shape), (exponent + 127).astype(np.uint8)


def dequantize(codes, scales):
    codes = np.asarray(codes, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    if codes.shape[-1] % 32 or scales.shape != (
        *codes.shape[:-1],
        codes.shape[-1] // 32,
    ):
        raise ValueError("MXFP8 code/scale shape mismatch")
    if np.any((codes & 127) == 127) or np.any(scales == 255):
        raise ValueError("MXFP8 NaN encoding")
    exponent, fraction = (codes >> 3) & 15, (codes & 7).astype(np.float32)
    elements = np.where(
        exponent == 0,
        fraction / np.float32(512),
        np.ldexp(8 + fraction, exponent.astype(np.int16) - 10),
    )
    elements = np.copysign(elements, np.where(codes & 128, -1.0, 1.0)).astype(
        np.float32
    )
    blocks = elements.reshape(*codes.shape[:-1], -1, 32)
    result = np.ldexp(blocks, scales.astype(np.int16)[..., None] - 127).reshape(
        codes.shape
    )
    if not np.isfinite(result).all():
        raise ValueError("MXFP8 decoded value overflows FP32")
    return result
