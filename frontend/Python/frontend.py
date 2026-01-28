# ===- frontend.py -------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# This is the entry of the Buddy Compiler frontend.
#
# TODO[Low]: When integrating more frameworks, `frontend.py` acts as a unified
# entry and driver, separating out compilers/importers for various platforms
# (e.g. DynamoCompiler).
#
# ===---------------------------------------------------------------------------

from typing import Any, List, Optional
import operator
import os
import ctypes
import platform
import functools
import numpy as np

import mlir.ir as ir
import mlir.dialects.func as func
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir import runtime as rt
import torch
import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified
import torch.utils._pytree as pytree

from .ops.linalg import ops_registry as linalg_ops_registry
from .ops.tosa import ops_registry as tosa_ops_registry
from .ops.math import ops_registry as math_ops_registry
from .ops.func import ops_registry as func_ops_registry
from .graph import Graph, TensorDType, TensorMeta
from .graph.operation import *
from .graph.transform import (
    affine_grid_generator_homogeneous_base_simplify,
    affine_grid_generator_simplify,
    replace_bernoulli_with_runtime_rng,
    replace_exponential_with_runtime_rng,
    replace_geometric_with_runtime_rng,
    replace_rand_with_runtime_rng,
    maxpool2d_simplify,
    functionalize_out_overloads,
)
from .graph.type import *


class DynamoCompiler:
    """
    Dynamo Compiler is one of the frontends of Buddy Compiler.
    Dynamo Compiler acts as a custom compiler for the TorchDynamo framework,
    which converts an FX Graph into an equivalent Buddy Graph and MLIR module.

    Attributes:
        imported_graphs: The imported graphs.
        imported_params: The imported parameters from the model.
    """

    @staticmethod
    def _sanitize_decompositions(
        decompositions: Optional[dict],
    ) -> Optional[dict]:
        """Patch aot_autograd decompositions to avoid decomposed graphs we cannot lower correctly."""
        if decompositions is None:
            return None
        if not isinstance(decompositions, dict):
            return decompositions

        patched = dict(decompositions)

        def _no_decomp(*args, **kwargs):
            return NotImplemented

        # Some Inductor decompositions rewrite 1D pooling into a
        # unsqueeze/avg_pool2d/squeeze pattern with negative dims that do not
        # round-trip after the rank change. Keep the original ATen ops so Buddy
        # can lower them directly.
        for key in (
            torch.ops.aten.avg_pool1d.default,
            torch.ops.aten.adaptive_avg_pool1d.default,
        ):
            if key in patched:
                patched[key] = _no_decomp

        return patched

    def __init__(
        self,
        func_name: str = "forward",
        primary_registry: Optional[dict] = None,
        aot_autograd_decomposition: Optional[dict] = None,
        verbose=False,
        enable_external_calls: bool = False,
        enable_out_functionalize: bool = False,
    ) -> None:
        """
        Initializes the Dynamo Compiler.

        Args:
            func_name: The function name to be used.
            primary_registry (dict, optional): The primary operations registry.
            aot_autograd_decomposition (Optional[dict], optional):
            The ahead-of-time autograd decomposition dictionary.
            verbose (bool): Controls whether to print additional information for
                debugging purposes. The default value is False, indicating that
                no extra debug information will be printed.
            enable_external_calls (bool): Enable external function call support (for oneDNN, etc.)
        Attributes:
            _func_name: The function name to be used.
            _aot_autograd_decomposition (Optional[dict], optional):
            The ahead-of-time autograd decomposition dictionary.
            _verbose: The option for the verbosity option of output.
            _imported_graphs: The buddy graphs from dynamo importer.
            _ops_registry (dict, optional): The buddy operations' lower func
            registry.
            _imported_params: The model params extract from torch.
            _ops_map: The torch aten ops map with buddy ops.

        """
        # Make custom dynamo compiler take effect.
        dynamo.reset()
        # Initialize the attributes.
        if primary_registry is None:
            primary_registry = {}
        self._func_name = func_name
        self._aot_autograd_decomposition = self._sanitize_decompositions(
            aot_autograd_decomposition
        )
        self._verbose = verbose
        self._enable_external_calls = enable_external_calls
        env_out = (
            os.getenv("BUDDY_ENABLE_OUT_FUNCTIONALIZE", "").strip().lower()
        )
        self._enable_out_functionalize = enable_out_functionalize or (
            env_out in ("1", "true", "yes", "y", "on")
        )
        self._imported_graphs = []
        self._ops_registry = {}
        self._imported_params = {}
        self._model_config = type("Config", (), {"decode_with_cache": False})
        self._ops_registry.update(math_ops_registry)
        self._ops_registry.update(linalg_ops_registry)
        self._ops_registry.update(tosa_ops_registry)
        self._ops_registry.update(func_ops_registry)
        self._ops_registry.update(primary_registry)
        self._ops_map = {
            "output": OutputOp,
            "placeholder": PlaceholderOp,
            "arange.start": ArangeOp,
            "arange.default": ArangeOp,
            "unsqueeze.default": UnsqueezeOp,
            "view.default": ViewOp,
            "view.dtype": ViewDtypeOp,
            "ones.default": OnesOp,
            "full.default": FullOp,
            "embedding.default": EmbeddingOp,
            "masked_fill.Scalar": MaskedFillOp,
            "slice.Tensor": SliceOp,
            "expand.default": ExpandOp,
            "_to_copy.default": ToCopyOp,
            "rsub.Scalar": RsubOp,
            "pow.Tensor_Scalar": PowOp,
            "mean.dim": MeanOp,
            "rsqrt.default": RsqrtOp,
            "mul.Tensor": MulOp,
            "mul.Scalar": MulOp,
            "t.default": TOp,
            "mm.default": MatmulOp,
            "transpose.int": TransposeOp,
            "index.Tensor": IndexOp,
            "neg.default": NegOp,
            "cat.default": CatOp,
            "bmm.default": BatchMatmulOp,
            "div.default": DivOp,
            "div.Tensor": DivOp,
            "div.Tensor_mode": DivTensorModeOp,
            "div.Tensor_mode": DivTensorModeOp,
            "_softmax.default": SoftmaxOp,
            "_log_softmax.default": LogSoftmaxOp,
            "_log_softmax.default": LogSoftmaxOp,
            "clone.default": CloneOp,
            "silu.default": SiluOp,
            "add.Tensor": AddOp,
            "addmm.default": AddMMOp,
            "permute.default": PermuteOp,
            "convert_element_type.default": ConvertElementTypeOp,
            "sum.dim_IntList": SumDimOp,
            "tanh.default": TanhOp,
            "sub.Tensor": SubOp,
            "var_mean.correction": VarMeanOp,
            "amax.default": AmaxOp,
            "select.int": SelectOp,
            "exp.default": ExpOp,
            "erf.default": ErfOp,
            "getitem": GetItemOp,
            "convolution.default": Conv2dOp,
            "max_pool2d_with_indices.default": MaxPool2dWithIndicesOp,
            "_low_memory_max_pool_with_offsets.default": MaxPool2dWithIndicesOp,
            "_low_memory_max_pool2d_with_offsets.default": MaxPool2dWithIndicesOp,
            "max_pool1d.default": MaxPool1dOp,
            "max_pool1d_with_indices.default": MaxPool1dOp,
            "max_pool3d.default": MaxPool3dOp,
            "max_pool3d_with_indices.default": MaxPool3dOp,
            "fractional_max_pool2d.default": FractionalMaxPool2dOp,
            "fractional_max_pool2d.output": FractionalMaxPool2dOp,
            "avg_pool1d.default": AvgPool1dOp,
            "avg_pool3d.default": AvgPool3dOp,
            "adaptive_max_pool1d.default": AdaptiveMaxPool1dOp,
            "adaptive_max_pool2d.default": AdaptiveMaxPool2dOp,
            "adaptive_avg_pool1d.default": AdaptiveAvgPool1dOp,
            "_adaptive_avg_pool2d.default": AdaptiveAvgPool2dOp,
            "_adaptive_avg_pool3d.default": AdaptiveAvgPool3dOp,
            "relu.default": ReluOp,
            "iota.default": IotaOp,
            "sigmoid.default": SigmoidOp,
            "glu.default": GluOp,
            "glu.out": GluOp,
            "gru.input": GruOp,
            "scalar_tensor.default": ScalarTensorOp,
            "where.self": WhereOp,
            "sqrt.default": SqrtOp,
            "abs.default": AbsOp,
            "log.default": LogOp,
            "ceil.default": CeilOp,
            "floor.default": FloorOp,
            "maximum.default": MaximumOp,
            "minimum.default": MinimumOp,
            "bitwise_not.default": BitwiseNotOp,
            "logical_not.default": LogicalNotOp,
            "clamp.default": ClampOp,
            "logical_and.default": LogicalAndOp,
            "logical_or.default": LogicalOrOp,
            "bitwise_or.Tensor": BitwiseOrOp,
            "bitwise_xor.Tensor": BitwiseXorOp,
            "bitwise_or.Scalar_Tensor": BitwiseOrOp,
            "bitwise_xor.Scalar_Tensor": BitwiseXorOp,
            "bitwise_left_shift.Tensor": BitwiseLeftShiftOp,
            "bitwise_left_shift.Tensor_Scalar": BitwiseLeftShiftOp,
            "bitwise_left_shift.Scalar_Tensor": BitwiseLeftShiftOp,
            "bitwise_right_shift.Tensor": BitwiseRightShiftOp,
            "bitwise_right_shift.Tensor_Scalar": BitwiseRightShiftOp,
            "bitwise_right_shift.Scalar_Tensor": BitwiseRightShiftOp,
            "amin.default": AminOp,
            "min.default": AminOp,
            "min.unary_out": AminOp,
            "avg_pool2d.default": AvgPool2dOp,
            "logical_xor.default": LogicalXorOp,
            "prod.default": ProdOp,
            "prod.dim_int": ProdOp,
            "eq.Tensor": EqTensorOp,
            "ne.Tensor": NeTensorOp,
            "gt.Tensor": GtTensorOp,
            "ge.Tensor": GeTensorOp,
            "lt.Tensor": LtTensorOp,
            "le.Tensor": LeTensorOp,
            "log10.default": Log10Op,
            "log2.default": Log2Op,
            "log1p.default": Log1pOp,
            "expm1.default": Expm1Op,
            "exponential.default": ExponentialOp,
            "exponential.out": ExponentialOp,
            "exponential_.default": ExponentialOp,
            "constant_pad_nd.default": ConstantPadNdOp,
            "reciprocal.default": ReciprocalOp,
            "clamp_min.default": ClampMinOp,
            "clamp_max.default": ClampMaxOp,
            "randint.low": RandIntLowOp,
            "cos.default": CosOp,
            "sin.default": SinOp,
            "argmax.default": ArgMaxOp,
            "split.Tensor": SplitOp,
            "max.default": MaxOp,
            "gt.Scalar": GtOp,
            "_scaled_dot_product_flash_attention_for_cpu.default": ScaledDotProductFlashAttentionForCpuOp,
            "ge.Scalar": GeOp,
            "gcd.default": GcdOp,
            "gcd.out": GcdOp,
            "gcd_.default": GcdOp,
            "kthvalue.default": KthValueOp,
            "kthvalue.values": KthValueOp,
            "kthvalue.dimname": KthValueOp,
            "kthvalue.dimname_out": KthValueOp,
            "_unsafe_index.Tensor": UnsafeIndexOp,
            "eq.Scalar": EqualOp,
            "copy.default": CopyOp,
            "slice_scatter.default": SliceScatterOp,
            "bitwise_and.Tensor": BitwiseAndTensorOp,
            "bitwise_and.Scalar_Tensor": BitwiseAndTensorOp,
            "bitwise_or.Scalar_Tensor_out": BitwiseOrOp,
            "bitwise_xor.Scalar_Tensor_out": BitwiseXorOp,
            "bitwise_left_shift.Tensor_out": BitwiseLeftShiftOp,
            "bitwise_left_shift.Tensor_Scalar_out": BitwiseLeftShiftOp,
            "bitwise_left_shift.Scalar_Tensor_out": BitwiseLeftShiftOp,
            "bitwise_right_shift.Tensor_out": BitwiseRightShiftOp,
            "bitwise_right_shift.Tensor_Scalar_out": BitwiseRightShiftOp,
            "bitwise_right_shift.Scalar_Tensor_out": BitwiseRightShiftOp,
            "index_put.default": IndexPutOp,
            "ne.Scalar": NeScalarOp,
            "cumsum.default": CumsumOp,
            "cumprod.default": CumProdOp,
            "sort.default": SortOp,
            "sort.stable": SortOp,
            "sort.values_stable": SortOp,
            "searchsorted.Tensor": SearchSortedOp,
            "searchsorted.Tensor_out": SearchSortedOp,
            "searchsorted.Scalar": SearchSortedOp,
            "searchsorted.Scalar_out": SearchSortedOp,
            "_tensor_constant": TensorConstantOp,
            "lift_fresh_copy.default": LiftFreshCopyOp,
            "repeat.default": RepeatOp,
            "repeat_interleave.Tensor": RepeatInterleaveOp,
            "repeat_interleave.Tensor_out": RepeatInterleaveOp,
            "repeat_interleave.self_Tensor": RepeatInterleaveOp,
            "repeat_interleave.self_int": RepeatInterleaveOp,
            "as_strided.default": AsStridedOp,
            "as_strided_copy.default": AsStridedOp,
            "as_strided_copy.out": AsStridedOp,
            "as_strided_scatter.default": AsStridedScatterOp,
            "as_strided_scatter.out": AsStridedScatterOp,
            "scatter.src": ScatterSrcOp,
            "scatter.value": ScatterValueOp,
            "scatter.reduce": ScatterReduceOp,
            "scatter.value_reduce": ScatterReduceOp,
            "scatter.reduce_out": ScatterReduceOp,
            "scatter.value_reduce_out": ScatterReduceOp,
            "scatter_.reduce": ScatterReduceOp,
            "scatter_.value_reduce": ScatterReduceOp,
            "scatter_reduce.two": ScatterReduceOp,
            "index_reduce.default": ScatterReduceOp,
            "index_reduce.out": ScatterReduceOp,
            "index_reduce_.default": ScatterReduceOp,
            "gather.default": GatherOp,
            "native_layer_norm.default": NativeLayerNormOp,
            "native_group_norm.default": NativeGroupNormOp,
            "_native_batch_norm_legit.default": NativeBatchNormLegitOp,
            "_native_batch_norm_legit.no_stats": NativeBatchNormLegitNoStatsOp,
            "_native_batch_norm_legit_no_training.default": NativeBatchNormLegitNoTrainingOp,
            "native_dropout.default": NativeDropoutOp,
            "upsample_bilinear2d.vec": UpsampleBilinear2dVecOp,
            "upsample_nearest2d.vec": UpsampleNearest2dVecOp,
            "grid_sampler_2d.default": GridSampler2dOp,
            "grid_sampler_3d.default": GridSampler3dOp,
            "grid_sampler_3d.out": GridSampler3dOp,
            "col2im.default": Col2imOp,
            "sym_size.int": SymSizeOp,
            "sym_stride.int": SymStrideOp,
            "sym_numel.default": SymNumelOp,
            "sym_storage_offset.default": SymStorageOffsetOp,
            "gelu.default": GeluOp,
            "flip.default": FlipOp,
            "leaky_relu.default": LeakyReluOp,
            "hardtanh.default": HardtanhOp,
            "elu.default": EluOp,
            "sign.default": SignOp,
            "signbit.default": SignbitOp,
            "signbit.out": SignbitOp,
            "round.default": RoundOp,
            "trunc.default": TruncOp,
            "sinh.default": SinhOp,
            "cosh.default": CoshOp,
            "tan.default": TanOp,
            "exp2.default": Exp2Op,
            "zeros.default": ZerosOp,
            "zeros_like.default": ZerosLikeOp,
            "ones_like.default": OnesLikeOp,
            "full_like.default": FullLikeOp,
            "all.default": AllOp,
            "all.dim": AllOp,
            "any.default": AnyOp,
            "any.dim": AnyOp,
            "isinf.default": IsInfOp,
            "isnan.default": IsNanOp,
            "floor_divide.default": FloorDivideOp,
            "fmod.Tensor": FmodOp,
            "fmod.Scalar": FmodOp,
            "remainder.Tensor": RemainderOp,
            "remainder.Scalar": RemainderOp,
            "remainder.Scalar_Tensor": RemainderOp,
            "remainder.Scalar_Tensor_out": RemainderOp,
            "pow.Tensor_Tensor": PowTensorTensorOp,
            "softplus.default": SoftplusOp,
            "hardswish.default": HardswishOp,
            "tile.default": TileOp,
            "stack.default": StackOp,
            "lerp.Tensor": LerpOp,
            "lerp.Scalar": LerpOp,
            "clamp.Tensor": ClampTensorOp,
            "le.Scalar": LeScalarOp,
            "lt.Scalar": LtScalarOp,
            "index_select.default": IndexSelectOp,
            "scatter_add.default": ScatterAddOp,
            "arange.start_step": ArangeStartStepOp,
            "min.dim": MinDimOp,
            "min.dim_min": MinDimOp,
            "argmin.default": ArgMinOp,
            "squeeze.default": SqueezeOp,
            "squeeze.dim": SqueezeDimOp,
            "squeeze.dims": SqueezeDimsOp,
            "unfold.default": UnfoldOp,
            "uniform.default": UniformOp,
            "uniform.out": UniformOp,
            "uniform_.default": UniformOp,
            "bernoulli.Tensor": BernoulliOp,
            "bernoulli.default": BernoulliOp,
            "topk.default": TopkOp,
            "unbind.int": UnbindOp,
            # Batched matrix operations
            "baddbmm.default": BaddbmmOp,
            # Special math functions
            "lgamma.default": LgammaOp,
            "digamma.default": DigammaOp,
            "logcumsumexp.default": LogCumsumExpOp,
            "logcumsumexp.out": LogCumsumExpOp,
            "logcumsumexp.dimname": LogCumsumExpOp,
            "logcumsumexp.dimname_out": LogCumsumExpOp,
            "frexp.Tensor": FrexpOp,
            "frexp.Tensor_out": FrexpOp,
            "histc.default": HistcOp,
            "histc.out": HistcOp,
            "igamma.default": IgammaOp,
            "igamma.out": IgammaOp,
            "igamma_.default": IgammaOp,
            "igammac.default": IgammacOp,
            "igammac.out": IgammacOp,
            "igammac_.default": IgammacOp,
            "median.default": MedianOp,
            "median.dim": MedianOp,
            "median.dim_values": MedianOp,
            "median.names_dim": MedianOp,
            "median.names_dim_values": MedianOp,
            "median.out": MedianOp,
            "nanmedian.default": NanMedianOp,
            "nanmedian.dim": NanMedianOp,
            "nanmedian.dim_values": NanMedianOp,
            "nanmedian.names_dim": NanMedianOp,
            "nanmedian.names_dim_values": NanMedianOp,
            "nanmedian.out": NanMedianOp,
            "mode.default": ModeOp,
            "mode.values": ModeOp,
            "mode.dimname": ModeOp,
            "mode.dimname_out": ModeOp,
            "new_empty_strided.default": NewEmptyStridedOp,
            "nonzero_static.default": NonzeroStaticOp,
            "min.other": MinimumOp,
            "min.out": MinimumOp,
            "i0.default": I0Op,
            "erfc.default": ErfcOp,
            "erfinv.default": ErfinvOp,
            "special_entr.default": SpecialEntrOp,
            "special_entr.out": SpecialEntrOp,
            "special_i0e.default": SpecialI0eOp,
            "special_i0e.out": SpecialI0eOp,
            "special_i1.default": SpecialI1Op,
            "special_i1.out": SpecialI1Op,
            "special_i1e.default": SpecialI1eOp,
            "special_i1e.out": SpecialI1eOp,
            "special_modified_bessel_i0.default": I0Op,
            "special_modified_bessel_i0.out": I0Op,
            "special_modified_bessel_i1.default": SpecialI1Op,
            "special_modified_bessel_i1.out": SpecialI1Op,
            "special_erfcx.default": SpecialErfcxOp,
            "special_erfcx.out": SpecialErfcxOp,
            "special_ndtr.default": SpecialNdtrOp,
            "special_ndtr.out": SpecialNdtrOp,
            "special_log_ndtr.default": SpecialLogNdtrOp,
            "special_log_ndtr.out": SpecialLogNdtrOp,
            "special_ndtri.default": SpecialNdtriOp,
            "special_ndtri.out": SpecialNdtriOp,
            "special_spherical_bessel_j0.default": SpecialSphericalBesselJ0Op,
            "special_spherical_bessel_j0.out": SpecialSphericalBesselJ0Op,
            # Cumulative operations
            "cummax.default": CummaxOp,
            "cummin.default": CumminOp,
            # Tensor clamp operations
            "clamp_min.Tensor": ClampMinTensorOp,
            "clamp_max.Tensor": ClampMaxTensorOp,
            # Additional elementwise operations
            "hypot.default": HypotOp,
            "copysign.Tensor": CopysignOp,
            "copysign.Scalar": CopysignOp,
            "copysign.Scalar_out": CopysignOp,
            "copysign_.Scalar": CopysignOp,
            "nextafter.default": NextafterOp,
            "masked_scatter.default": MaskedScatterOp,
            "rev.default": RevOp,
            # Scalar arithmetic operations
            "add.Scalar": AddScalarOp,
            "sub.Scalar": SubScalarOp,
            "div.Scalar": DivScalarOp,
            "div.Scalar_mode": DivScalarModeOp,
            "pow.Scalar": PowScalarOp,
            # Reduction operations
            "mean.default": MeanDefaultOp,
            "var.correction": VarCorrectionOp,
            "var.dim": VarDimOp,
            "any.dims": AnyDimsOp,
            # Trigonometric functions
            "acos.default": AcosOp,
            "asin.default": AsinOp,
            "atan.default": AtanOp,
            "atan2.default": Atan2Op,
            "acosh.default": AcoshOp,
            "asinh.default": AsinhOp,
            "atanh.default": AtanhOp,
            # Other operations
            "fill.Scalar": FillScalarOp,
            "diagonal.default": DiagonalOp,
            "diagonal_scatter.default": DiagonalScatterOp,
            "diagonal_scatter.out": DiagonalScatterOp,
            "alias.default": AliasOp,
            "empty.memory_format": EmptyOp,
            "rand.default": RandOp,
            "randn.default": RandnOp,
            "geometric.default": GeometricOp,
            "geometric.out": GeometricOp,
            "geometric_.default": GeometricOp,
            "select_scatter.default": SelectScatterOp,
            "select_scatter.out": SelectScatterOp,
            "split_with_sizes.default": SplitWithSizesOp,
            "max.dim": MaxDimOp,
            "nonzero.default": NonzeroOp,
            # Standard deviation operations
            "std.default": StdDefaultOp,
            "std.dim": StdDimOp,
            "std.correction": StdCorrectionOp,
            # Additional reduction operations
            "sum.default": SumDefaultOp,
            "all.dims": AllDimsOp,
            "var.default": VarDefaultOp,
            # Norm operations
            "norm.Scalar": NormScalarOp,
            "norm.ScalarOpt_dim": NormScalarOptDimOp,
            # Backward operations (Gradient Computation)
            "_adaptive_avg_pool2d_backward.default": AdaptiveAvgPool2dBackwardOp,
            "avg_pool2d_backward.default": AvgPool2dBackwardOp,
            "convolution_backward.default": ConvolutionBackwardOp,
            "embedding_dense_backward.default": EmbeddingDenseBackwardOp,
            "max_pool2d_with_indices_backward.default": MaxPool2dWithIndicesBackwardOp,
            "native_group_norm_backward.default": NativeGroupNormBackwardOp,
            "native_layer_norm_backward.default": NativeLayerNormBackwardOp,
            # Bitwise scalar operations
            "bitwise_and.Scalar": BitwiseAndScalarOp,
            "bitwise_or.Scalar": BitwiseOrScalarOp,
            "bitwise_xor.Scalar": BitwiseXorScalarOp,
            "bitwise_and.Scalar_Tensor_out": BitwiseAndTensorOp,
            "bitwise_and_.Tensor": BitwiseAndTensorOp,
            "bitwise_and_.Scalar": BitwiseAndScalarOp,
            "bitwise_or_.Tensor": BitwiseOrOp,
            "bitwise_or_.Scalar": BitwiseOrScalarOp,
            "bitwise_xor_.Tensor": BitwiseXorOp,
            "bitwise_xor_.Scalar": BitwiseXorScalarOp,
            "bitwise_left_shift_.Tensor": BitwiseLeftShiftOp,
            "bitwise_left_shift_.Tensor_Scalar": BitwiseLeftShiftOp,
            "bitwise_right_shift_.Tensor": BitwiseRightShiftOp,
            "bitwise_right_shift_.Tensor_Scalar": BitwiseRightShiftOp,
            # Padding operations
            "reflection_pad1d.default": ReflectionPad1dOp,
            "reflection_pad2d.default": ReflectionPad2dOp,
            "reflection_pad3d.default": ReflectionPad3dOp,
            "replication_pad2d.default": ReplicationPad2dOp,
            "replication_pad3d.default": ReplicationPad3dOp,
            # Other missing core aten operations
            "empty_strided.default": EmptyStridedOp,
            "randperm.default": RandpermOp,
            # Core Aten remaining operations
            "_embedding_bag.default": EmbeddingBagOp,
            "_embedding_bag_forward_only.default": EmbeddingBagOp,
            "_cdist_forward.default": CdistForwardOp,
            "_pdist_forward.default": PdistForwardOp,
            "_fft_r2c.default": FftR2cOp,
            "_local_scalar_dense.default": LocalScalarDenseOp,
            "resize_.default": ResizeOp,
            "resize.default": ResizeOp,
        }

    @property
    def imported_graphs(self):
        """Returns the imported buddy graphs after compilation."""
        return self._imported_graphs

    @property
    def imported_params(self):
        """Returns the imported model params after compilation."""
        return self._imported_params

    def _torch_dtype_translate(self, dtype):
        match dtype:
            case "torch.int64":
                return TensorDType.Int64
            case "torch.int32":
                return TensorDType.Int32
            case "torch.int8":
                return TensorDType.Int8
            case "torch.float16":
                return TensorDType.Float16
            case "torch.bfloat16":
                return TensorDType.BFloat16
            case "torch.float32":
                return TensorDType.Float32
            case "torch.float64":
                return TensorDType.Float64
            case "torch.complex64":
                return TensorDType.Complex64
            case "torch.complex128":
                return TensorDType.Complex128
            case "torch.bool":
                return TensorDType.Bool
            case _:
                raise NotImplementedError(f"Unsupported dtype: {dtype}")

    def _create_node(
        self,
        gm_node_name: str,
        node_name: str,
        node_input: Tuple,
        node_users: List[str],
        node_output_shape: list = [],
        node_output_dtype: TensorDType = None,
        node_kwargs: Optional[Dict] = None,
    ):
        """
        Create buddy op node from torch aten op.

        Args:
            gm_node_name: The op node class map to buddy op by _ops_map.
            node_name: The op node name to be used.
            node_input: The args input to op node.
            node_output_shape: The list of the op node's output shape.
            node_output_dtype: The TensorDType enum type of the op node's output
            data type.
            node_kwargs: The restful attributes for op node.
        """
        op_class = self._ops_map[gm_node_name]
        buddy_node = op_class()
        buddy_node._name = node_name
        if gm_node_name == "output":
            for input_arg in node_input[0]:
                if input_arg is None:
                    continue
                buddy_node.add_argument(str(input_arg))
            return buddy_node

        def _add_arg_and_parents(arg):
            if isinstance(arg, torch.fx.Node):
                buddy_node.add_argument(str(arg))
                buddy_node.add_parent(str(arg))
            elif isinstance(arg, torch.dtype):
                buddy_node.add_argument(self._torch_dtype_translate(str(arg)))
            elif isinstance(arg, (list, tuple)):
                # Traverse elements to collect parent nodes but keep the container as a single argument
                for item in arg:
                    if isinstance(item, torch.fx.Node):
                        buddy_node.add_parent(str(item))
                buddy_node.add_argument(arg)
            else:
                buddy_node.add_argument(arg)
            return arg

        for input_arg in node_input:
            _add_arg_and_parents(input_arg)
        for user in node_users:
            buddy_node.add_children(user)
        if node_kwargs is None:
            node_kwargs = {}
        buddy_node._keyword_arguments.update(node_kwargs)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

    def _compile_fx(
        self,
        gm: torch.fx.GraphModule,
        inputs: List[torch.Tensor],
        *,
        return_type: str = "eager",
    ) -> Any:
        """
        Compiles the provided FX Graph to Buddy Graph.

        Args:
            gm (torch.fx.GraphModule): The GraphModule to be compiled.
            inputs (List[torch.Tensor]): The input tensors.
            return_type (str): Controls the compiled callable that AOTAutograd
                receives from the Buddy compiler.
                - "eager": return the FX graph forward (legacy behavior).
                - "buddy": return a Buddy MLIR execution callable.

        Returns:
            dynamo_run: The function of the ahead-of-time compiled module,
            return for torchdynamo's call.
        """

        # params = {
        #     # **dict(gm.named_parameters(remove_duplicate=False)),
        #     **dict(gm.named_buffers(remove_duplicate=False)),
        # }
        # print(len(params))
        # params_flat, _ = pytree.tree_flatten(params)
        inputs_pos = []
        params_pos = []
        buffers_pos = []
        for i, node in enumerate(gm.graph.nodes):
            if i >= len(inputs):
                break
            if not str(node).startswith("l_self"):
                inputs_pos.append(i)
            elif "buffer" in str(node):
                buffers_pos.append(i)
            else:
                params_pos.append(i)

        params_flat = [inputs[i] for i in params_pos + buffers_pos]

        if self._verbose:
            print("Graph in tabular form:")
            gm.graph.print_tabular()

        def _compiler(_gm: torch.fx.GraphModule, _inputs: List[torch.Tensor]):
            """Compile a FX graph in Aten/Prims IR to MLIR."""
            num_cached_kv = 0
            if self._model_config.decode_with_cache:
                num_cached_kv = self._model_config.num_hidden_layers * 2
            func_inputs = []
            for i in inputs_pos:
                # for inp in _inputs[len(params_flat) :]:
                inp = _inputs[i + num_cached_kv]
                inp_shape = inp.shape
                inp_dtype = self._torch_dtype_translate(str(inp.dtype))
                func_inputs.append(TensorMeta(inp_shape, inp_dtype))
            for inp in _inputs[:num_cached_kv]:
                inp = _inputs[i]
                inp_shape = inp.shape
                inp_dtype = self._torch_dtype_translate(str(inp.dtype))
                func_inputs.append(TensorMeta(inp_shape, inp_dtype))
            fake_params = []
            for param in params_flat:
                param_dtype = self._torch_dtype_translate(str(param.dtype))
                fake_params.append(TensorMeta(param.shape, param_dtype))
            graph = Graph(
                func_inputs,
                fake_params,
                self._ops_registry,
                self._func_name,
                DeviceType.CPU,
                self._verbose,
                self._enable_external_calls,
            )
            graph._params_ref = params_flat
            param_nodes = []
            buffers_nodes = []
            input_nodes = []
            other_nodes = []
            for i, node in enumerate(
                list(_gm.graph.nodes)[num_cached_kv:], start=0
            ):
                if i in params_pos:
                    param_nodes.append(node)
                elif i in buffers_pos:
                    buffers_nodes.append(node)
                elif i in inputs_pos:
                    input_nodes.append(node)
                else:
                    other_nodes.append(node)
            input_nodes.extend(list(_gm.graph.nodes)[:num_cached_kv])
            gm_nodes = param_nodes + buffers_nodes + input_nodes + other_nodes

            for gm_node in gm_nodes:
                node_users = []
                for user in gm_node.users.keys():
                    node_users.append(str(user))
                if gm_node.op == "placeholder":
                    node_dtype = self._torch_dtype_translate(
                        str(gm_node.meta["tensor_meta"].dtype)
                    )
                    buddy_node = self._create_node(
                        gm_node.op,
                        gm_node.name,
                        gm_node.args,
                        node_users,
                        gm_node.meta["tensor_meta"].shape,
                        node_dtype,
                    )

                elif gm_node.op == "output":
                    buddy_node = self._create_node(
                        gm_node.op, gm_node.name, gm_node.args, node_users
                    )

                elif gm_node.target is operator.getitem:
                    node_dtype = self._torch_dtype_translate(
                        str(gm_node.meta["tensor_meta"].dtype)
                    )
                    buddy_node = self._create_node(
                        str(gm_node.target.__name__),
                        gm_node.name,
                        gm_node.args,
                        node_users,
                        gm_node.meta["tensor_meta"].shape,
                        node_dtype,
                    )
                elif gm_node.op == "get_attr":
                    if "_tensor_constant" in gm_node.name:
                        import re

                        stack_trace = gm_node.meta.get("stack_trace") or ""
                        match = re.search(
                            r"torch\.tensor\(([-+]?\d+(\.\d+)?), dtype=[a-zA-Z]+\)",
                            stack_trace,
                        )
                        value = None
                        if match:
                            value = float(match.group(1))
                        if value is None:
                            val = gm_node.meta.get("val")
                            if isinstance(val, torch.Tensor):
                                if val.numel() != 1:
                                    raise NotImplementedError(
                                        "_tensor_constant only supports scalar tensors"
                                    )
                                value = val.item()
                            elif isinstance(val, (int, float)):
                                value = val
                        if value is None:
                            raise NotImplementedError(
                                "Unsupported _tensor_constant format"
                            )
                        gm_node.insert_arg(len(gm_node.args), value)
                        val = gm_node.meta.get("val")
                        node_shape = val.shape
                        node_dtype = self._torch_dtype_translate(str(val.dtype))
                        buddy_node = self._create_node(
                            "_tensor_constant",
                            gm_node.name,
                            gm_node.args,
                            node_users,
                            node_shape,
                            node_dtype,
                            node_kwargs=gm_node.kwargs,
                        )
                else:
                    tensor_meta = gm_node.meta.get("tensor_meta")
                    val = gm_node.meta.get("val")
                    # num_returns = len(gm_node.target._schema.returns)
                    num_returns = (
                        len(val)
                        if isinstance(val, (list, tuple))
                        else len(gm_node.target._schema.returns)
                    )
                    if num_returns == 1:
                        if tensor_meta is None:
                            if isinstance(val, torch.Tensor):
                                node_dtype = self._torch_dtype_translate(
                                    str(val.dtype)
                                )
                                node_shape = val.shape
                            elif str(gm_node.target) == "aten.unbind.int":
                                input_node = gm_node.args[0]
                                dim = (
                                    gm_node.args[1]
                                    if len(gm_node.args) > 1
                                    else 0
                                )
                                input_meta = input_node.meta.get("tensor_meta")
                                input_val = input_node.meta.get("val")
                                if input_meta is None:
                                    if not isinstance(input_val, torch.Tensor):
                                        raise RuntimeError(
                                            "Missing input meta for aten.unbind.int"
                                        )
                                    input_shape = list(input_val.shape)
                                    input_dtype = input_val.dtype
                                else:
                                    input_shape = list(input_meta.shape)
                                    input_dtype = input_meta.dtype
                                if dim < 0:
                                    dim += len(input_shape)
                                length = input_shape[dim]
                                if length < 0:
                                    raise RuntimeError(
                                        "Dynamic unbind dimension not supported"
                                    )
                                out_shape = tuple(
                                    input_shape[:dim] + input_shape[dim + 1 :]
                                )
                                node_shape = tuple([out_shape] * length)
                                node_dtype = tuple(
                                    [
                                        self._torch_dtype_translate(
                                            str(input_dtype)
                                        )
                                    ]
                                    * length
                                )
                            else:
                                raise RuntimeError(
                                    f"Missing tensor_meta for {gm_node.target}"
                                )
                        else:
                            node_dtype = self._torch_dtype_translate(
                                str(tensor_meta.dtype)
                            )
                            node_shape = tensor_meta.shape
                    elif num_returns > 1:
                        node_dtype = tuple(
                            [
                                self._torch_dtype_translate(str(val_item.dtype))
                                for val_item in val
                            ]
                        )
                        node_shape = tuple([val_item.shape for val_item in val])
                    else:
                        raise RuntimeError("Zero returns is not supported.")

                    buddy_node = self._create_node(
                        str(gm_node.target.__name__),
                        gm_node.name,
                        gm_node.args,
                        node_users,
                        node_shape,
                        node_dtype,
                        node_kwargs=gm_node.kwargs,
                    )
                    buddy_node._torch_op = str(gm_node.target.__name__)
                    out_kwarg_names: List[str] = []
                    schema = getattr(gm_node.target, "_schema", None)
                    if schema is not None:
                        for arg in getattr(schema, "arguments", ()):
                            if getattr(arg, "is_out", False):
                                t = str(getattr(arg, "type", "")).lower()
                                if "tensor" in t:
                                    out_kwarg_names.append(arg.name)
                    buddy_node._torch_out_kwarg_names = out_kwarg_names
                graph.add_node(buddy_node)
            transform_list = [
                maxpool2d_simplify,
                affine_grid_generator_simplify,
                affine_grid_generator_homogeneous_base_simplify,
            ]
            if self._enable_external_calls:
                transform_list.extend(
                    [
                        replace_bernoulli_with_runtime_rng,
                        replace_exponential_with_runtime_rng,
                        replace_geometric_with_runtime_rng,
                        replace_rand_with_runtime_rng,
                    ]
                )
            if self._enable_out_functionalize:
                transform_list.insert(
                    0,
                    functools.partial(
                        functionalize_out_overloads, ops_map=self._ops_map
                    ),
                )
            graph.perform(transform_list)
            self._imported_graphs.append(graph)
            self._imported_params[graph] = params_flat
            if return_type == "eager":
                return _gm.forward
            if return_type == "buddy":
                exec_list = self._dynamo_run_for_graph(graph)

                def _exec(*args):
                    outs = exec_list(*args)
                    if len(outs) == 1:
                        return outs[0]
                    return tuple(outs)

                return _exec
            raise ValueError(
                f"Unsupported return_type={return_type!r}; expected 'eager' or 'buddy'."
            )

        return aot_module_simplified(
            gm,
            inputs,
            fw_compiler=_compiler,
            decompositions=self._aot_autograd_decomposition,
        )

    def __call__(
        self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]
    ) -> Any:
        """
        A callable method that wraps around the `_compile_fx` method.

        Args:
            gm (torch.fx.GraphModule): The GraphModule to be compiled.
            inputs (List[torch.Tensor]): The input tensors.

        Returns:
            dynamo_run: The function of the ahead-of-time compiled module,
            return for torchdynamo's call.
        """
        return self._compile_fx(gm, inputs)

    def importer(self, model, *args, **kwargs) -> List[Graph]:
        """
        Imports the provided model as MLIR module and flat parameters.

        Args:
            model: The model to be imported.
            args: Arguments for the model.
            kwargs: Keyword arguments for the model.

        Returns:
            imported_graphs: The imported buddy graphs.
        """
        if hasattr(model, "config") and model.config is not None:
            self._model_config = model.config.__class__.from_dict(
                model.config.to_dict()
            )
        if (
            "use_cache" in kwargs
            and kwargs["use_cache"]
            and "past_key_values" in kwargs
        ):
            self._model_config.decode_with_cache = True
        else:
            self._model_config.decode_with_cache = False
        model_opt = dynamo.optimize(self._compile_fx)(model)
        model_opt(*args, **kwargs)
        return self._imported_graphs

    def importer_by_export(
        self, module: torch.nn.Module, *args, **kwargs
    ) -> List[Graph]:
        """
        Imports the provided model as MLIR module and flat parameters by `torch.export.export`.
        The previous `importer` method use the dynamo API, which may cause the imported FX graph
        have input arguments in a different order from the original PyTorch model. See also:

        -  [PyTorch Export API](https://docs.pytorch.org/docs/stable/export.html)
        -  [PyTorch Issue #128334](https://github.com/pytorch/pytorch/issues/128334)

        Args:
            module: `torch.nn.Module` The model to be imported.
            args: Arguments for the model.
            kwargs: Keyword arguments for the model.

        Returns:
            imported_graphs: The imported buddy graphs.
        """
        exported_program = torch.export.export(module, args, kwargs)
        self._compile_fx(exported_program.graph_module, list(args))
        return self._imported_graphs

    def _dynamo_run_for_graph(self, graph):
        """
        Build an execution callable for a specific Buddy graph.
        """

        def get_lib_extension():
            if platform.system() == "Linux":
                return ".so"
            elif platform.system() == "Darwin":
                return ".dylib"
            else:
                raise RuntimeError("Unsupported platform")

        graph.compile()
        # Collect dependency libraries.
        lib_extension = get_lib_extension()
        lib_names = ["libmlir_runner_utils", "libmlir_c_runner_utils", "libomp"]
        path_prefix = os.path.dirname(os.path.abspath(__file__))
        lib_base_path = os.path.join(path_prefix, "../../../../llvm/build/lib/")
        lib_base_path = os.path.abspath(lib_base_path)
        shared_libs = [
            os.path.join(lib_base_path, lib_name + lib_extension)
            for lib_name in lib_names
        ]
        buddy_lib_base_path = os.path.abspath(
            os.path.join(path_prefix, "../../../lib")
        )
        buddy_rng_lib = os.path.join(
            buddy_lib_base_path, "libbuddy_rng_utils" + lib_extension
        )
        if self._enable_external_calls:
            shared_libs.append(buddy_rng_lib)
        # Define execution engine.
        ee = ExecutionEngine(
            graph._imported_module, opt_level=3, shared_libs=shared_libs
        )

        def exec_buddy_graph(*args):
            """
            Execute a graph using TorchDynamo with the provided input tensors.

            Args:
                *args: List[torch.Tensor]
                Input tensors to be passed to the graph's function.

            Returns:
            List[torch.Tensor]
                The result of executing the graph, represented as a list of
                output tensors.
            """

            def _bf16_tensor_to_numpy_uint16(
                tensor: torch.Tensor,
            ) -> np.ndarray:
                """
                Convert a CPU bfloat16 tensor to a NumPy uint16 array containing
                raw BF16 bit patterns (to avoid torch.bfloat16 -> numpy errors).
                """
                if tensor.device.type != "cpu":
                    tensor = tensor.cpu()
                # BF16 stores the top 16 bits of float32. Casting BF16->F32 is exact.
                f32 = tensor.to(dtype=torch.float32).contiguous().numpy()
                u32 = f32.view(np.uint32)
                return (u32 >> 16).astype(np.uint16, copy=False)

            def _bf16_uint16_numpy_to_f32(npy: np.ndarray) -> np.ndarray:
                """
                Convert a NumPy uint16 array (BF16 bit patterns) to float32 NumPy.
                """
                u32 = np.asarray(npy, dtype=np.uint32) << 16
                return u32.view(np.float32)

            def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
                if tensor.device.type != "cpu":
                    tensor = tensor.cpu()
                tensor = tensor.contiguous()
                if tensor.dtype == torch.bfloat16:
                    return _bf16_tensor_to_numpy_uint16(tensor)
                return tensor.numpy()

            input_arrays = []
            input_descs = []
            input_slots = []
            for tensor in args:
                npy = _tensor_to_numpy(tensor)
                npy = np.array(npy, copy=True)
                input_arrays.append(npy)
                desc = rt.get_ranked_memref_descriptor(npy)
                input_descs.append(desc)
                input_slots.append(ctypes.pointer(ctypes.pointer(desc)))

            output_struct = graph._output_descriptor()
            output_ptr = ctypes.pointer(output_struct)
            output_slot = ctypes.pointer(output_ptr)

            ee.invoke(graph._func_name, output_slot, *input_slots)

            output_tensors = []
            for i in range(len(graph._output_memref)):
                out_desc = getattr(output_struct, str(i))
                out = rt.ranked_memref_to_numpy(ctypes.pointer(out_desc))
                if isinstance(out, np.ndarray) and out.dtype == np.uint16:
                    out = _bf16_uint16_numpy_to_f32(out)
                if isinstance(out, np.ndarray):
                    output_tensors.append(torch.from_numpy(out))
                else:
                    output_tensors.append(torch.tensor(out))
            return output_tensors

        return exec_buddy_graph

    def dynamo_run(self):
        """
        A callable method that wraps around the `exec_buddy_graph` method.

        Returns:
            exec_buddy_graph: The function of the ahead-of-time compiled module,
            return for torchdynamo's call.
        """
        # Dynamo's graph break may import more than one graph.
        graph = self._imported_graphs[-1]
        return self._dynamo_run_for_graph(graph)


class TorchCompileBackend:
    """
    TorchDynamo backend wrapper for `torch.compile(backend=...)`.

    The backend callable signature is:
        backend(gm, example_inputs) -> callable
    """

    def __init__(self, compiler: DynamoCompiler) -> None:
        self._compiler = compiler

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ):
        # Keep per-compile state bounded; torch.compile caches the returned
        # callable per-graph, so this is safe for common use.
        self._compiler._imported_graphs = []
        self._compiler._imported_params = {}
        return self._compiler._compile_fx(
            gm, list(example_inputs), return_type="buddy"
        )


def make_default_torch_backend() -> TorchCompileBackend:
    try:
        import torch._inductor.lowering  # noqa: F401
    except Exception:
        # Some builds do not eagerly expose `torch._inductor.lowering` as an
        # attribute. Inductor decompositions may access it via
        # `torch._inductor.lowering`, so import the submodule explicitly to
        # avoid runtime AttributeError.
        pass

    from torch._inductor.decomposition import decompositions as inductor_decomp

    compiler = DynamoCompiler(
        primary_registry=tosa_ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    return TorchCompileBackend(compiler)


# Public default backend instance for `torch.compile(backend=...)`.
dynamo_compiler = make_default_torch_backend()
