# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
import torch._dynamo as dynamo
from torch._inductor.decomposition import decompositions as inductor_decomp
import numpy

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa
from buddy.compiler.graph.transform import eliminate_transpose


def test_weight_transpose_data_correctness():
    """
    Test that when a weight transpose is eliminated, the weight data is correctly
    transposed when stored, ensuring E2E correctness.
    """

    # Create a simple model where weight is a model parameter
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Weight shape: [4, 3], will be transposed to [3, 4] before use
            self.weight = torch.nn.Parameter(
                torch.zeros([4, 3], dtype=torch.float32)
            )
            # Initialize with known pattern: weight[i, j] = i * 10 + j
            with torch.no_grad():
                for i in range(4):
                    for j in range(3):
                        self.weight[i, j] = i * 10.0 + j

        def forward(self, x):
            # Weight shape: [4, 3], transpose to [3, 4]
            # x is [1, 3], so matmul(x, weight.T) = matmul([1,3], [3,4]) = [1,4]
            transposed_weight = torch.transpose(self.weight, 0, 1)
            return torch.matmul(x, transposed_weight)

    # Create model and input
    model = SimpleModel()
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)  # [1, 3]

    # Expected transposed weight: [3, 4] where weight_t[j, i] = i * 10 + j
    expected_weight_t = model.weight.T.clone()

    # Initialize the dynamo compiler
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )

    # Import model - this will extract model parameters
    graphs = dynamo_compiler.importer(model, x)
    assert len(graphs) == 1
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]

    # Debug: print all params before optimization
    print(f"Total params before optimization: {len(params)}")
    for idx, param in enumerate(params):
        print(f"  Param {idx}: shape={param.shape}, dtype={param.dtype}")

    # Apply the optimization to eliminate weight transpose
    graph.perform([eliminate_transpose])

    # Check that transpose info was recorded
    assert hasattr(
        graph, "_transposed_params"
    ), "transposed_params should be recorded"
    print(f"Transposed params indices: {list(graph._transposed_params.keys())}")

    # Find weight parameter - params list contains original weights with original shapes
    # So we should look for [4, 3] (original shape)
    weight_param_idx = None
    for idx, param in enumerate(params):
        # Check original shape [4, 3]
        if param.shape == torch.Size([4, 3]):
            weight_param_idx = idx
            print(
                f"  Found weight parameter at index {idx} with original shape [4, 3]"
            )
            break

    # Also check if there's any param that could be the weight
    if weight_param_idx is None:
        print("  Warning: Could not find weight with shape [4, 3]")
        print("  Available param shapes:", [p.shape for p in params])
        # Try to find by matching the first dimension or checking all params
        for idx, param in enumerate(params):
            if len(param.shape) == 2 and (
                param.shape[0] == 4 or param.shape[1] == 3
            ):
                weight_param_idx = idx
                print(
                    f"  Using param {idx} with shape {param.shape} as potential weight"
                )
                break

    assert (
        weight_param_idx is not None
    ), f"Should find weight parameter. Available params: {[p.shape for p in params]}"
    assert (
        weight_param_idx in graph._transposed_params
    ), f"Weight param {weight_param_idx} should be in _transposed_params"

    transpose_info = graph._transposed_params[weight_param_idx]
    print(f"  Transpose info: {transpose_info}")
    # torch.transpose can be 'transpose', 't', or 'permute' depending on how it's lowered
    assert transpose_info["type"] in [
        "transpose",
        "t",
        "permute",
    ], f"Should be transpose, t, or permute type, got {transpose_info['type']}"
    # Check dims/perm - for 't' type it's [1, 0], for 'transpose' it's [0, 1], for 'permute' it's [1, 0]
    if transpose_info["type"] == "transpose":
        assert transpose_info["dims"] == [
            0,
            1,
        ], f"Should swap dimensions 0 and 1, got {transpose_info['dims']}"
    elif transpose_info["type"] == "t":
        assert transpose_info["dims"] == [
            1,
            0,
        ], f"For t type, dims should be [1, 0], got {transpose_info['dims']}"
    elif transpose_info["type"] == "permute":
        assert transpose_info["perm"] == [
            1,
            0,
        ], f"For permute type, perm should be [1, 0], got {transpose_info['perm']}"

    # Get original weight
    original_weight_np = params[weight_param_idx].detach().numpy()

    # Apply transpose to the weight data as would be done in import script
    if transpose_info["type"] == "t":
        # 2D transpose [1, 0]
        stored_weight_np = original_weight_np.T
    elif transpose_info["type"] == "transpose":
        # Swap two dimensions
        dim1, dim2 = transpose_info["dims"]
        stored_weight_np = numpy.swapaxes(original_weight_np, dim1, dim2)
    elif transpose_info["type"] == "permute":
        # Apply permutation
        stored_weight_np = numpy.transpose(
            original_weight_np, transpose_info["perm"]
        )
    else:
        raise ValueError(f"Unknown transpose type: {transpose_info['type']}")

    # Get expected transposed weight for comparison
    expected_weight_np = expected_weight_t.detach().numpy()

    # Print original weight values for CHECK
    print("Original Weight (before transpose):")
    print(f"Shape: {original_weight_np.shape}")
    for i in range(original_weight_np.shape[0]):
        row_values = [
            f"{original_weight_np[i, j]:.1f}"
            for j in range(original_weight_np.shape[1])
        ]
        print(f"Row {i}: {' '.join(row_values)}")

    # Print stored weight values (after transpose) for CHECK
    print("\nStored Weight (after transpose, as stored in arg0.data):")
    print(f"Shape: {stored_weight_np.shape}")
    for i in range(stored_weight_np.shape[0]):
        row_values = [
            f"{stored_weight_np[i, j]:.1f}"
            for j in range(stored_weight_np.shape[1])
        ]
        print(f"Row {i}: {' '.join(row_values)}")

    # Verify the stored weight matches expected
    assert numpy.allclose(stored_weight_np, expected_weight_np), (
        f"Stored weight data should match expected transposed weight.\n"
        f"Stored shape: {stored_weight_np.shape}, Expected shape: {expected_weight_np.shape}\n"
        f"Max difference: {numpy.max(numpy.abs(stored_weight_np - expected_weight_np))}"
    )

    print("\n✓ Weight transpose data correctness test passed")


if __name__ == "__main__":
    test_weight_transpose_data_correctness()

# CHECK: Original Weight (before transpose):
# CHECK: Shape: (4, 3)
# CHECK: Row 0: 0.0 1.0 2.0
# CHECK: Row 1: 10.0 11.0 12.0
# CHECK: Row 2: 20.0 21.0 22.0
# CHECK: Row 3: 30.0 31.0 32.0
# CHECK: Stored Weight (after transpose, as stored in arg0.data):
# CHECK: Shape: (3, 4)
# CHECK: Row 0: 0.0 10.0 20.0 30.0
# CHECK: Row 1: 1.0 11.0 21.0 31.0
# CHECK: Row 2: 2.0 12.0 22.0 32.0
