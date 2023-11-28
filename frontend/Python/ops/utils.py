from mlir import ir
import torch
def _torch_dtype_str_to_mlir_dtype(dtype: str) -> ir.Type:
    """
    Converts a torch dtype to the corresponding MLIR dtype.

    Args:
        dtype (torch.dtype): The torch data type.

    Returns:
        mlir.ir.Type: The corresponding MLIR data type.

    Raises:
        NotImplementedError: If the given dtype is not supported.
    """
    match dtype:
        case 'torch.int32':
            return ir.IntegerType.get_signless(32)
        case 'torch.int64':
            return ir.IntegerType.get_signless(64)
        case 'torch.float32':
            return ir.F32Type.get()
        case 'torch.bool':
            return ir.IntegerType.get_signless(1)
        case _:
            raise NotImplementedError(f"Unsupported dtype {dtype}")

def _torch_dtype_to_mlir_dtype(dtype: torch.dtype) -> ir.Type:
    """
    Converts a torch dtype to the corresponding MLIR dtype.

    Args:
        dtype (torch.dtype): The torch data type.

    Returns:
        mlir.ir.Type: The corresponding MLIR data type.

    Raises:
        NotImplementedError: If the given dtype is not supported.
    """
    match dtype:
        case torch.int32:
            return ir.IntegerType.get_signless(32)
        case torch.int64:
            return ir.IntegerType.get_signless(64)
        case torch.float32:
            return ir.F32Type.get()
        case torch.bool:
            return ir.IntegerType.get_signless(1)
        case _:
            raise NotImplementedError(f"Unsupported dtype {dtype}")