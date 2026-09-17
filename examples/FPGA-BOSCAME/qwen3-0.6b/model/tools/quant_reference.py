"""Host numerical oracle for the existing Triton W8A8 contract (not inference).

Only independent quantize/dequantize/linear building blocks live here. The
model's forward ordering must come from its imported Buddy graph.
"""
import numpy as np


def quantize_rows(value):
    """Symmetric signed [-127,127], per last-axis row, half away from zero.

    Operations including division/rounding are FP32, matching tl.div_rn and
    sign-dependent 0.5 adjustment then truncation in the shared Triton kernel.
    Zero rows use scale 1; nonfinite values and underflowed scales are errors.
    """
    value=np.asarray(value,dtype=np.float32)
    if value.ndim<1 or value.shape[-1]==0 or not np.isfinite(value).all():
        raise ValueError("quantization requires nonempty finite rows")
    maximum=np.max(np.abs(value),axis=-1,keepdims=True)
    scale=np.where(maximum==0,np.float32(1),maximum/np.float32(127)).astype(np.float32)
    if np.any(scale==0):raise ValueError("FP32 quantization scale underflow")
    divided=value/scale
    adjusted=divided+np.where(divided>=0,np.float32(0.5),np.float32(-0.5))
    quantized=np.trunc(np.clip(adjusted,np.float32(-127),np.float32(127))).astype(np.int8)
    return quantized,np.squeeze(scale,axis=-1)


def linear_w8a8(activation, weight_i8, weight_scale):
    """A[M,K], physical W[N,K] -> fresh F32 output[M,N], no bias.

    AME accumulation starts from zero; quantized input residual paths are never
    reused as accumulators. Residual and normalization belong to FP32 graph ops.
    """
    activation=np.asarray(activation,dtype=np.float32)
    weight=np.asarray(weight_i8)
    scale=np.asarray(weight_scale,dtype=np.float32)
    if activation.ndim!=2 or weight.ndim!=2 or weight.dtype!=np.int8 or activation.shape[1]!=weight.shape[1]:
        raise ValueError("expected A[M,K] and signed int8 W[N,K]")
    if scale.shape!=(weight.shape[0],) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError("expected positive finite per-output-channel weight scales")
    if np.any(weight==-128) or weight.shape[1]*127*127>np.iinfo(np.int32).max:
        raise ValueError("weight outside contract or int32 accumulation may overflow")
    quantized,activation_scale=quantize_rows(activation)
    accumulator=quantized.astype(np.int32) @ weight.astype(np.int32).T
    # Match shared Triton dequantize exactly: (acc_f32 * row_scale) * col_scale.
    # Reassociation can change FP32 rounding and is not part of this contract.
    return (accumulator.astype(np.float32)*activation_scale[:,None])*scale[None,:]


def error_metrics(actual,reference):
    actual=np.asarray(actual,dtype=np.float64)
    reference=np.asarray(reference,dtype=np.float64)
    if actual.shape!=reference.shape or actual.size==0 or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("metrics require equal nonempty finite arrays")
    error=np.abs(actual-reference)
    return {"max_abs_error":float(error.max()),"mean_abs_error":float(error.mean())}
