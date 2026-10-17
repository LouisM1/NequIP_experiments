"""Experiment 14: independent scalar readouts on a frozen NequIP backbone.

Keep this module on the Python path when training or loading model.pt.
The two columns of the linear readout are independent heads; the scaler
selects only the column corresponding to the central atom's element.
"""
import math

import torch
from e3nn import o3
from graph_pes.atomic_graph import AtomicGraph
from graph_pes.utils.nn import PerElementParameter

class ElementHeadScaler(torch.nn.Module):
    """Route two scalar outputs to O/Si and apply fixed training-set scales."""

    def __init__(self, si_std: float, o_std: float):
        super().__init__()
        # Deliberately not a LocalTensorScaler: the existing scalar pre-fit
        # adapter would overwrite that class's fixed scales with subset stats.
        self.per_element_scaling = PerElementParameter.of_length(1, requires_grad=False)
        self.register_buffer("head_index", torch.full((119,), -1, dtype=torch.long))
        self.head_index[8] = 0
        self.head_index[14] = 1
        with torch.no_grad():
            self.per_element_scaling.zero_()
            self.per_element_scaling[8] = o_std
            self.per_element_scaling[14] = si_std
        self.per_element_scaling.requires_grad_(False)
        self.per_element_scaling.register_elements([8, 14])

    def forward(self, outputs: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        if outputs.dim() != 2 or outputs.shape[1] != 2:
            raise ValueError("Expected two scalar readout columns: O, then Si")
        indices = self.head_index[graph.Z]
        if bool((indices < 0).any()):
            raise ValueError("Experiment 14 supports only oxygen and silicon")
        selected = outputs.gather(1, indices.unsqueeze(1)).squeeze(1)
        return selected * self.per_element_scaling[graph.Z].reshape(-1)

    def pre_fit(self, graphs: AtomicGraph):
        # Fixed, full-training-split statistics are supplied by the config.
        pass

    def non_decayable_parameters(self):
        return [self.per_element_scaling]


def fresh_element_heads(model, si_std, o_std):
    """Attach independently initialised O/Si heads after E/F-only transfer.

    No NMR checkpoint is loaded. The configured experiment seed controls the
    two random e3nn linear weight columns. Offsets are initialised by the
    normal training-only scalar pre-fit; fixed scales are preserved there.
    Only the 256 readout weights and two active offsets are trainable.
    """
    if not all(math.isfinite(s) and s > 0 for s in (si_std, o_std)):
        raise ValueError("Training standard deviations must be positive and finite")
    if model._has_been_pre_fit.item():
        raise ValueError("Fresh heads require an un-prefitted destination model")
    nequip = model.models["nequip"]
    old = nequip.tensor_readout
    if (str(old.irreps_in) != "128x0e" or str(old.irreps_out) != "1x0e"
            or old.weight.numel() != 128 or old.bias.numel() != 0
            or nequip.number_of_tps is not None):
        raise ValueError("Expected a direct readout from 128 scalar features")

    # e3nn independently samples all 256 weights; no existing head is copied.
    new = o3.Linear(old.irreps_in, "2x0e").to(old.weight)
    scaler = ElementHeadScaler(si_std, o_std).to(old.weight)
    with torch.no_grad():
        model.models["offset"]._offsets.zero_()

    nequip.tensor_readout = new
    nequip.scaler = scaler
    model.requires_grad_(False)
    new.requires_grad_(True)
    model.models["offset"]._offsets.requires_grad_(True)
    for parameter in model.parameters():
        if isinstance(parameter, PerElementParameter):
            parameter.register_elements([8, 14])
    print("Fresh independent O/Si heads; 258 active trainable parameters. "
          "E/F backbone and training-set scales are fixed; no NMR weights loaded.",
          flush=True)
    return model
