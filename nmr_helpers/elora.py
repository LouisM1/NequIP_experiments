"""Path-wise ELoRA for the existing scalar NequIP transfer experiment.

The original e3nn operation still performs the contraction, preserving its
coupling paths, parity and normalization. Only its channel weights change.
"""
import math

import torch
from e3nn import o3


class PathELoRA(torch.nn.Module):
    """Wrap an e3nn Linear or FullyConnectedTensorProduct with W0 + BA."""

    def __init__(self, base, rank):
        super().__init__()
        if type(rank) is not int or rank < 1:
            raise ValueError("ELoRA rank must be a positive integer")
        if not isinstance(base, (o3.Linear, o3.FullyConnectedTensorProduct)):
            raise TypeError("ELoRA expects Linear or FullyConnectedTensorProduct")
        if not base.internal_weights or not base.shared_weights:
            raise ValueError("ELoRA expects fixed, internally stored channel weights")
        if isinstance(base, o3.Linear) and base.bias_numel:
            raise ValueError("This experiment uses bias-free equivariant linears")

        self.base = base.requires_grad_(False)
        self.A = torch.nn.ParameterList()
        self.B = torch.nn.ParameterList()
        self.is_tensor_product = isinstance(base, o3.FullyConnectedTensorProduct)
        for weight in base.weight_views():
            # e3nn stores (input, output) or (input1, input2, output).
            n_in = math.prod(weight.shape[:-1])
            n_out = weight.shape[-1]
            if rank > min(n_in, n_out):
                raise ValueError(f"Rank {rank} exceeds channel block {tuple(weight.shape)}")
            a = torch.nn.Parameter(weight.new_empty(rank, n_in))
            b = torch.nn.Parameter(weight.new_zeros(n_out, rank))
            torch.nn.init.normal_(a, std=1.0 / math.sqrt(n_in))
            self.A.append(a)
            self.B.append(b)

    def forward(self, x, y=None):
        # One independent BA per allowed path, not one factorization of the
        # flattened weight vector. Transpose back to e3nn's storage order.
        delta = torch.cat([(b @ a).T.reshape(-1) for a, b in zip(self.A, self.B)])
        weight = self.base.weight + delta  # multiplier 1; no extra alpha/r
        if self.is_tensor_product:
            return self.base(x, y, weight=weight)
        return self.base(x, weight=weight)


def add_elora(model, rank=4):
    """Adapt three operations in each of the four transferred layers.

    Call after from_energy_force_model. Train the existing NMR head, scales,
    offsets and new factors; freeze every original backbone parameter.
    """
    net = model.models["nequip"]
    if len(net.layers) != 4 or str(net.target_tensor_irreps) != "1x0e":
        raise ValueError("Expected the four-layer scalar TensorNequIP experiment")
    model.requires_grad_(False)
    net.tensor_readout.requires_grad_(True)
    net.scaler.requires_grad_(True)
    model.models["offset"].requires_grad_(True)

    # Construction occurs on CPU, after the seeded NMR head is initialized.
    # Adapter initialization must not shift the baseline's subsequent RNG state.
    with torch.random.fork_rng(devices=[]):
        for layer in net.layers:
            layer.pre_message_linear = PathELoRA(layer.pre_message_linear, rank)
            layer.post_message_linear = PathELoRA(layer.post_message_linear, rank)
            layer.self_interaction.tensor_product = PathELoRA(
                layer.self_interaction.tensor_product, rank
            )
    print(f"Added rank-{rank} ELoRA to 12 operations across all four layers; "
          "training adapters and the NMR head/scales/offsets.", flush=True)
    return model
