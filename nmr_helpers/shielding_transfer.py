"""Transfer a compatible graph-pes NequIP backbone to scalar TensorNequIP.

The checkpoint must be your own trusted graph-pes model.pt file.
Energy readout, energy offsets/scales, optimizer state and pre-fit flags
are deliberately not transferred.
"""
from pathlib import Path
import hashlib
import torch
from graph_pes.models import NequIP, TensorNequIP


def from_energy_force_model(model, checkpoint, freeze_backbone=False):
    """Copy the E/F backbone; optionally freeze it for head-only transfer.

    The fresh shielding readout, offsets and scales stay trainable.
    The default preserves experiment 09's full-model fine-tuning.
    """
    path = Path(checkpoint)
    source = torch.load(path, map_location='cpu', weights_only=False)
    old = source.models['nequip']
    new = model.models['nequip']
    if not isinstance(old, NequIP) or not isinstance(new, TensorNequIP):
        raise TypeError('Expected a NequIP source and TensorNequIP destination')
    if str(new.target_tensor_irreps) != '1x0e':
        raise ValueError('This transfer helper is for scalar isotropic shielding')
    if not torch.allclose(old.cutoff.cpu(), new.cutoff.cpu()):
        raise ValueError('Source and destination cutoffs differ')
    if len(old.layers) != len(new.layers):
        raise ValueError('Source and destination layer counts differ')
    if not torch.equal(old.Z_embedding.Z_to_idx.cpu(), new.Z_embedding.Z_to_idx.cpu()):
        raise ValueError('Source and destination element ordering differs')
    for a, b in zip(old.layers, new.layers):
        if a.irreps_in != b.irreps_in or a.irreps_out != b.irreps_out:
            raise ValueError('Source and destination layer irreps differ')
        if repr(a.aggregation) != repr(b.aggregation):
            raise ValueError('Source and destination aggregation differs')
    if repr(old.edge_embedding) != repr(new.edge_embedding):
        raise ValueError('Source and destination edge representations differ')

    # Strict per-module copying fails on missing or mismatched weights.
    parts = ('Z_embedding', 'initial_node_embedding', 'edge_embedding', 'layers')
    for name in parts:
        getattr(new, name).load_state_dict(getattr(old, name).state_dict(), strict=True)
    model.requires_grad_(True)
    if freeze_backbone:
        for name in parts:
            getattr(new, name).requires_grad_(False)
    model.train()
    if model._has_been_pre_fit.item():
        raise ValueError('Destination must be a fresh, un-prefitted shielding model')
    print(f'Transferred NequIP embeddings and {len(new.layers)} layers from {path}', flush=True)
    print(f'Source SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}', flush=True)
    if freeze_backbone:
        print('Frozen embeddings and interaction layers; only fresh shielding '
              'readout, offsets and scales trainable.', flush=True)
    else:
        print('Fresh shielding readout, offsets and scales; all layers trainable.', flush=True)
    return model
