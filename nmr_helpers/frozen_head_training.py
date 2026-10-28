"""Cache invariant NequIP features; train the original head with AdamW."""
from pathlib import Path
import hashlib
import json
import os
import uuid
from importlib.metadata import version

import torch
from pytorch_lightning import Callback
from graph_pes.atomic_graph import AtomicGraph, neighbour_distances, neighbour_vectors
from graph_pes.models import TensorNequIP
from graph_pes.data.transforms import identity_transform
from graph_pes.training.opt import Optimizer

FEATURE_KEY = 'nmr_frozen_node_features'
BACKBONE_PARTS = ('Z_embedding', 'initial_node_embedding', 'edge_embedding', 'layers')


class CachedTensorNequIP(TensorNequIP):
    """Identical scalar model, with an optional cached input to its readout.

    Graphs without cached features use the full backbone, including when an
    exported model is loaded for prediction. No new parameters are introduced.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.target_method != 'direct' or str(self.target_tensor_irreps) != '1x0e':
            raise ValueError('Frozen-feature training supports the scalar direct head only.')

    def frozen_features(self, graph: AtomicGraph) -> torch.Tensor:
        r = neighbour_distances(graph)
        Y = self.edge_embedding(neighbour_vectors(graph))
        Z_embed = self.Z_embedding(graph.Z)
        node_embed = self.initial_node_embedding(graph.Z)
        for layer in self.layers:
            node_embed = layer(node_embed, Z_embed, r, Y, graph)
        return node_embed

    def forward(self, graph: AtomicGraph) -> dict[str, torch.Tensor]:
        if 'nmr_frozen_node_features' in graph.other:
            node_embed = graph.other['nmr_frozen_node_features']
        else:
            node_embed = self.frozen_features(graph)
        return {'tensor': self.scaler(self.tensor_readout(node_embed), graph)}


def _hash_tensor(digest, name, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())


def backbone_digest(net):
    digest = hashlib.sha256()
    for part in BACKBONE_PARTS:
        module = getattr(net, part)
        if any(p.requires_grad for p in module.parameters()):
            raise ValueError(f'Cannot cache a trainable backbone: {part}')
        digest.update(repr(module).encode())
        for name, tensor in sorted(module.state_dict().items()):
            _hash_tensor(digest, part + '.' + name, tensor)
    _hash_tensor(digest, 'cutoff', net.cutoff)
    return digest.hexdigest()


def cache_dataset(dataset, net, directory):
    """Attach detached per-atom features without changing labels or ordering."""
    if dataset.transform not in (None, identity_transform):
        raise ValueError('Frozen-feature caching requires data without transforms.')
    graphs = list(dataset.graphs)
    device = next(net.parameters()).device
    metadata = dict(schema=1, backbone=backbone_digest(net), graph_pes=version('graph-pes'),
                    torch=str(torch.__version__), dtype=str(next(net.parameters()).dtype),
                    device=str(device.type), matmul_precision=torch.get_float32_matmul_precision(),
                    cuda_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cuda_device=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    structures=len(graphs))
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for i, graph in enumerate(graphs):
        # Labels are deliberately excluded: cached features depend only on
        # the frozen model and graph, and labels are always read afresh.
        for name in graph._fields:
            if name in ('properties', 'other'):
                continue
            value = getattr(graph, name)
            if torch.is_tensor(value):
                _hash_tensor(digest, f'{i}.{name}', value)
            else:
                digest.update(str((i, name, value)).encode())
    metadata['key'] = digest.hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (metadata['key'] + '.pt')
    hit = path.is_file()
    if hit:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['metadata'] != metadata:
            raise ValueError(f'Cache metadata mismatch: {path}')
        features = payload['features']
    else:
        features = []
        with torch.no_grad():
            for graph in graphs:
                features.append(net.frozen_features(graph.to(device)).detach().cpu().clone())
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            torch.save(dict(metadata=metadata, features=features), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if len(features) != len(graphs):
        raise ValueError(f'Cache structure count mismatch: {path}')
    for graph, h in zip(graphs, features):
        if (h.shape != (len(graph.Z), net.tensor_readout.irreps_in.dim)
                or h.dtype != graph.R.dtype or h.requires_grad or not torch.isfinite(h).all()):
            raise ValueError(f'Invalid cached features: {path}')
    dataset.graphs = [g._replace(other={**g.other, FEATURE_KEY: h}) for g, h in zip(graphs, features)]
    print(f'Frozen features: {"reused" if hit else "built"} {len(graphs)} structures ({path})', flush=True)
    return dict(path=str(path), hit=hit, structures=len(graphs))


class FrozenFeatureCache(Callback):
    """Cache train and validation data after pre-fit/checkpoint restoration."""
    def __init__(self, directory='graph-pes-results/.frozen-feature-cache'):
        self.directory = directory
        self.report = []

    def on_train_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError('Frozen-feature pilot supports one device per run.')
        net = pl_module.model.models['nequip']
        if not isinstance(net, CachedTensorNequIP):
            raise TypeError('FrozenFeatureCache requires CachedTensorNequIP.')
        valid = trainer.val_dataloaders
        if not isinstance(valid, (list, tuple)):
            valid = [valid]
        loaders = [trainer.train_dataloader, *valid]
        for loader in loaders:
            if loader.num_workers != 0:
                raise ValueError('Frozen-feature pilot requires num_workers=0.')
            self.report.append(cache_dataset(loader.dataset, net, self.directory))


class FrozenHeadOptimizer(Optimizer):
    """Keep AdamW/decay unchanged; set only the elemental scales' own LR."""
    def __init__(self, name='AdamW', lr=0.003, scale_lr=0.003, weight_decay=0.01, **kwargs):
        if name != 'AdamW' or scale_lr <= 0:
            raise ValueError('Pilot requires AdamW and a positive scales learning rate.')
        super().__init__(name, lr=lr, weight_decay=weight_decay, **kwargs)
        self.scale_lr = scale_lr

    def __call__(self, model):
        net = model.models['nequip']
        backbone_digest(net)  # fail before fitting if any backbone is trainable
        scale = net.scaler.per_element_scaling
        non_decay = {id(p) for p in model.non_decayable_parameters()}
        if id(scale) not in non_decay:
            raise ValueError('Elemental scales must be excluded from weight decay.')
        groups = []
        named = sorted(model.named_parameters())
        for label in ('non-decayable', 'scales', 'normal'):
            members = [(n, p) for n, p in named if
                       ('scales' if p is scale else 'non-decayable' if id(p) in non_decay else 'normal') == label]
            if members:
                group = dict(name=label, params=[p for _, p in members],
                             param_names=[n for n, _ in members])
                if label != 'normal':
                    group['weight_decay'] = 0.0
                if label == 'scales':
                    group['lr'] = self.scale_lr
                groups.append(group)
        return self.optimizer_class(groups, **self.kwargs)
