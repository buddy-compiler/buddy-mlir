"""
Two-layer MLP training via PyTorch + Buddy MLIR JIT.

Three modes are demonstrated:
  1. Pure PyTorch (baseline)
  2. Buddy MLIR forward-only  (compile_backward=False): forward graph compiled
     to MLIR, backward runs through PyTorch autograd.
  3. Buddy MLIR forward+backward (compile_backward=True): both graphs compiled
     to MLIR through Buddy's JIT pipeline.
"""

import torch
import torch.nn as nn
from buddy.compiler.frontend import DynamoCompiler, TorchCompileBackend
from buddy.compiler.ops import tosa
from model import MLP
from torch._inductor.decomposition import decompositions as inductor_decomp


def make_buddy_backend(compile_backward: bool) -> TorchCompileBackend:
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
        compile_backward=compile_backward,
    )
    return TorchCompileBackend(compiler)


def train(
    label: str,
    backend=None,
    input_dim: int = 16,
    hidden_dim: int = 32,
    output_dim: int = 4,
    batch_size: int = 8,
    epochs: int = 20,
    lr: float = 0.05,
    seed: int = 42,
):
    torch.manual_seed(seed)
    model = MLP(input_dim, hidden_dim, output_dim)

    if backend is not None:
        compiled_model = torch.compile(model, backend=backend, dynamic=False)
    else:
        compiled_model = model

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== {label} ===")
    w_before = model.fc1.weight.data.clone()
    initial_loss = None

    for epoch in range(epochs):
        torch.manual_seed(epoch)
        x = torch.randn(batch_size, input_dim)
        labels = torch.randint(0, output_dim, (batch_size,))

        optimizer.zero_grad()
        outputs = compiled_model(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch + 1:3d}/{epochs}  loss={loss.item():.4f}")

    w_after = model.fc1.weight.data
    weights_changed = not torch.equal(w_before, w_after)
    loss_decreased = loss.item() < initial_loss

    print(f"  weights changed : {weights_changed}")
    print(
        f"  loss decreased  : {loss_decreased}  "
        f"({initial_loss:.4f} → {loss.item():.4f})"
    )
    return loss.item()


def main():
    # ── 1. Baseline: pure PyTorch ──────────────────────────────────────────
    train("Pure PyTorch (baseline)", backend=None)

    # ── 2. Buddy MLIR JIT – forward only ──────────────────────────────────
    # Forward graph is compiled by Buddy MLIR; backward uses PyTorch autograd.
    fw_backend = make_buddy_backend(compile_backward=False)
    train(
        "Buddy MLIR JIT – forward only (compile_backward=False)",
        backend=fw_backend,
    )

    # ── 3. Buddy MLIR JIT – forward + backward ────────────────────────────
    # Both forward and backward graphs are compiled by Buddy MLIR.
    fwbw_backend = make_buddy_backend(compile_backward=True)
    train(
        "Buddy MLIR JIT – forward + backward (compile_backward=True)",
        backend=fwbw_backend,
    )

    print("\nAll training runs completed.")


if __name__ == "__main__":
    main()
