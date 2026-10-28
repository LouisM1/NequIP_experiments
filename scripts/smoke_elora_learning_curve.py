"""Check nested subsets and run two CPU optimizer steps per ELoRA config.

Run from the repository: .venv/bin/python scripts/smoke_elora_learning_curve.py
Production YAMLs and historical runs are never modified. Results default to /tmp.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--configs', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
ROOT = args.root.resolve()
CONFIGS = (args.configs or ROOT/'configs/22-ELoRA-LearningCurve').resolve()
OUT = (args.output or Path(tempfile.mkdtemp(prefix='elora-learning-curve-smoke-'))).resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/elora-smoke-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/elora-smoke-cache')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import torch
import yaml
import numpy as np
from graph_pes.data.datasets import file_dataset
from graph_pes.training.opt import Optimizer
from graph_pes.training.callbacks import SaveBestModel
from graph_pes.utils.nn import count_used_parameters
from pytorch_lightning import Callback
import graph_pes.scripts.train as training
from nmr_helpers.elora import PathELoRA
from nmr_helpers.shared_head_training import train_from_config
from nmr_helpers.training_fixes import install_training_fixes

torch.set_num_threads(4)
torch.set_default_dtype(torch.float32)
install_training_fixes()
launcher = (ROOT/'train.sbatch').read_text()
exec(launcher[launcher.index('def ordered_optimizer'):launcher.index('if sys.argv[1] !=')])
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
sizes = [10, 50, 100, 200, 800]
base_path = ROOT/'configs/22-ELoRA-LearningCurve/22-ELoRA-N100-R4.yaml'
base = yaml.safe_load(base_path.read_text())
original_hash = sha(base_path)
configs, indices = {}, {}

for n in sizes:
    path = CONFIGS/f'22-ELoRA-N{n}-R4.yaml'
    raw = yaml.safe_load(path.read_text())
    expected = deepcopy(base)
    expected['data']['train']['n'] = n
    expected['wandb']['name'] = f'22-ELoRA-N{n}-R4'
    assert raw == expected, f'Unexpected production change for N={n}'
    prior = yaml.safe_load((ROOT/f'configs/21-FrozenBackbone-CachedHead-LearningCurve/21-FrozenBackbone-N{n}-ScaleLR.yaml').read_text())
    assert raw['data'] == prior['data'], f'Data settings differ from experiment 21 for N={n}'
    data = file_dataset(**raw['data']['train'], cutoff=6., pre_transform=False)
    indices[n] = [int(i) for i in data.graphs.structures.indices]
    assert len(data) == len(set(indices[n])) == n
    assert len(data.graphs.structures.collection) == 800
    if n != sizes[0]:
        prev = sizes[sizes.index(n)-1]
        assert indices[n][:prev] == indices[prev], f'Non-nested N={prev} -> N={n}'
    configs[n] = raw
assert set(indices[800]) == set(range(800))
print('PASS: actual file_dataset selections are nested and match experiment 21.', flush=True)

def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b

class Audit(Callback):
    def on_train_start(self, trainer, task):
        dataset = trainer.train_dataloader.dataset
        assert len(dataset) == n
        assert list(dataset.graphs.structures.indices) == indices[n]
        val = trainer.val_dataloaders
        assert len((val[0] if isinstance(val, (list,tuple)) else val).dataset) == 50
        assert count_used_parameters(task.model, only_learnable=True) == 29060
        self.wrappers = {name: m for name,m in task.model.named_modules() if isinstance(m, PathELoRA)}
        assert len(self.wrappers) == 12
        self.before = {name:p.detach().clone() for name,p in task.model.named_parameters()}
        for name,p in task.model.named_parameters():
            adapter = any(name.startswith(prefix+'.A.') or name.startswith(prefix+'.B.') for prefix in self.wrappers)
            head = name.startswith(('models.nequip.tensor_readout.', 'models.nequip.scaler.', 'models.offset.'))
            assert p.requires_grad == (adapter or head), name
        self.steps = 0

    def on_before_optimizer_step(self, trainer, task, optimizer):
        for name,m in self.wrappers.items():
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in [*m.A,*m.B]), name
            assert any(torch.count_nonzero(p.grad) for p in m.B), name
            if self.steps:
                assert any(torch.count_nonzero(p.grad) for p in m.A), name
        self.steps += 1

    def on_train_end(self, trainer, task):
        assert self.steps == trainer.global_step == 2
        after = dict(task.model.named_parameters())
        for name,p in after.items():
            if not p.requires_grad:
                assert torch.equal(p, self.before[name]), name
        for prefix in self.wrappers:
            for factor in ['A', 'B']:
                assert any(not torch.equal(p, self.before[name]) for name,p in after.items() if name.startswith(prefix+'.'+factor+'.')), (prefix,factor)
        assert any(not torch.equal(p,self.before[name]) for name,p in after.items() if name.startswith('models.nequip.tensor_readout.'))
        for split in ['train','valid']:
            assert torch.isfinite(trainer.callback_metrics[f'{split}/loss/total'])
        self.valid_loss = float(trainer.callback_metrics['valid/loss/total'])

factory_original = training.trainer_from_config
test_original = training.test_with_lightning
reports = []
for n, config in configs.items():
    probe = deepcopy(config)
    probe['wandb'] = None
    probe['general'].update(root_dir=str(OUT), run_id=f'N{n}')
    # Smoke-only limits: retain full dataset membership, but make two updates.
    probe['fitting']['max_n_pre_fit'] = 4
    probe['fitting']['trainer_kwargs'].update(accelerator='cpu', max_epochs=1,
        limit_train_batches=8, limit_val_batches=2, enable_model_summary=False)
    for split in ['train','valid']:
        probe['data'][split]['pre_transform'] = False
    audit = Audit()
    def factory(config, output_dir):
        trainer = factory_original(config, output_dir)
        trainer.callbacks.append(audit)
        for cb in trainer.callbacks:
            if isinstance(cb, SaveBestModel):
                cb.try_to_deploy = False  # Skip unrelated LAMMPS deployment.
        return trainer
    training.trainer_from_config = factory
    training.test_with_lightning = lambda *a, **kw: None  # Skip full post-fit scoring.
    try:
        train_from_config(probe)
    finally:
        training.trainer_from_config = factory_original
        training.test_with_lightning = test_original
    directory = OUT/f'N{n}'
    best = torch.load(directory/'checkpoints/best.ckpt', map_location='cpu', weights_only=False)
    last = torch.load(directory/'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
    model = torch.load(directory/'model.pt', map_location='cpu', weights_only=False).eval()
    weights = {k.removeprefix('model.'):v for k,v in best['state_dict'].items() if k.startswith('model.')}
    assert same(model.state_dict(), weights)
    assert last['global_step'] == 2
    graph = file_dataset(**config['data']['valid'], cutoff=6., pre_transform=False)[0]
    with torch.no_grad():
        prediction = model(graph)['tensor']
    assert prediction.shape == graph.Z.shape and torch.isfinite(prediction).all()
    reports.append(dict(n=n, optimizer_steps=2, trainable_parameters=29060,
        validation_loss=audit.valid_loss, config_sha256=sha(CONFIGS/f'22-ELoRA-N{n}-R4.yaml'),
        frozen_parameters_unchanged=True, all_12_adapters_updated=True,
        head_updated=True, checkpoint_export_matches=True, exported_predictions_finite=True))
    print(f'PASS N={n}: two optimizer steps, finite validation, unchanged frozen weights, export reload.', flush=True)
assert sha(base_path) == original_hash
report = dict(passed=True, runs=reports, nested=True, matches_experiment_21=True,
    source_file=str(ROOT/'data/nmr_iso/train.xyz'), train_file_sha256=sha(ROOT/'data/nmr_iso/train.xyz'),
    subset_indices_zero_based=indices, torch=torch.__version__, numpy=np.__version__,
    smoke_overrides=dict(device='cpu', epochs=1, train_batches=8, validation_batches=2,
        max_n_pre_fit=4, graph_pre_transform=False, wandb=False, post_fit_scoring=False),
    production_settings_unchanged_except_n_and_name=True)
(OUT/'smoke-results.json').write_text(json.dumps(report, indent=2)+'\n')
print('ALL FIVE CONFIGS PASSED:', OUT/'smoke-results.json', flush=True)
