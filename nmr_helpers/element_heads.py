"""Experiment 14: independent scalar readouts on a frozen NequIP backbone.

Keep this module on the Python path when training or loading model.pt.
The two columns of the linear readout are independent heads; the scaler
selects only the column corresponding to the central atom's element.
"""
import torch
from e3nn import o3
from graph_pes.atomic_graph import AtomicGraph
from graph_pes.models.components.scaling import LocalTensorScaler
from graph_pes.utils.nn import PerElementParameter

class ElementHeadScaler(LocalTensorScaler):
    """Route O/Si outputs, retaining the existing trainable elemental scales."""

    def __init__(self):
        super().__init__(length=1)
        self.register_buffer("head_index", torch.full((119,), -1, dtype=torch.long))
        self.head_index[8] = 0
        self.head_index[14] = 1
        with torch.no_grad():
            self.per_element_scaling.fill_(1.0)
        self.per_element_scaling.register_elements([8, 14])

    def forward(self, outputs: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        if outputs.dim() != 2 or outputs.shape[1] != 2:
            raise ValueError("Expected two scalar readout columns: O, then Si")
        indices = self.head_index[graph.Z]
        if bool((indices < 0).any()):
            raise ValueError("Experiment 14 supports only oxygen and silicon")
        selected = outputs.gather(1, indices.unsqueeze(1)).squeeze(1)
        return selected * self.per_element_scaling[graph.Z].reshape(-1)

def fresh_element_heads(model):
    """Attach independently initialised O/Si heads after E/F-only transfer.

    No NMR checkpoint is loaded. The configured experiment seed controls the
    two random e3nn linear weight columns. Offsets are initialised by the
    normal training-only scalar pre-fit, which also initialises trainable
    scales from the training standard deviations, as in earlier experiments.
    256 readout weights, two scales and two offsets are trainable: 260 total.
    """
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
    scaler = ElementHeadScaler().to(old.weight)
    with torch.no_grad():
        model.models["offset"]._offsets.zero_()

    nequip.tensor_readout = new
    nequip.scaler = scaler
    model.requires_grad_(False)
    new.requires_grad_(True)
    scaler.requires_grad_(True)
    model.models["offset"]._offsets.requires_grad_(True)
    for parameter in model.parameters():
        if isinstance(parameter, PerElementParameter):
            parameter.register_elements([8, 14])
    print("Fresh independent O/Si heads; 260 active trainable parameters. "
          "E/F backbone frozen; elemental scales/offsets trainable. No NMR weights loaded.",
          flush=True)
    return model
