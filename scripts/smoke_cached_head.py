"""CPU smoke checks for both cached N100 pilots; never edits real run outputs.

Run from the repository: .venv/bin/python scripts/smoke_cached_head.py
Optional --root/--staged support reviewing changes before installing them.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nequip-cached-head-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/nequip-cached-head-cache')
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--staged', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
ROOT = args.root.resolve()
STAGE = (args.staged or ROOT).resolve()
OUT = (args.output or Path(tempfile.mkdtemp(prefix='nequip-cached-head-'))).resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import nmr_helpers
nmr_helpers.__path__.insert(0, str(STAGE / 'nmr_helpers'))
import torch
import yaml
from graph_pes.atomic_graph import to_batch
from graph_pes.config.shared import instantiate_config_from_dict, parse_model, parse_loss
from graph_pes.config.training import TrainingConfig
from graph_pes.data.loader import GraphDataLoader
from graph_pes.data.datasets import GraphDataset
from graph_pes.models import TensorNequIP
from e3nn.util.jit import script as e3nn_script
from graph_pes.scripts.utils import configure_general_options
from graph_pes.training.opt import Optimizer
from graph_pes.training.tasks import TrainingTask, get_all_eval_metrics
from graph_pes.utils.logger import log_to_file
from graph_pes.utils.nn import count_used_parameters, PerElementParameter
from graph_pes.training.callbacks import EarlyStoppingWithLogging, SaveBestModel
from pytorch_lightning import Callback
import graph_pes.scripts.train as training
from nmr_helpers.shared_head_training import train_from_config
from nmr_helpers.training_fixes import install_training_fixes
from nmr_helpers.frozen_head_training import FrozenFeatureCache, FEATURE_KEY, cache_dataset, backbone_digest

torch.set_num_threads(4)
install_training_fixes()
launcher = (STAGE / 'train.sbatch').read_text()
exec(launcher[launcher.index('def ordered_optimizer'):launcher.index('if sys.argv[1] !=')])
paths = sorted((STAGE / 'configs/20-FrozenBackbone-CachedHead-Pilot').glob('*.yaml'))
assert len(paths) == 2
configs = [yaml.safe_load(p.read_text()) for p in paths]
base = yaml.safe_load((ROOT / 'configs/18-FrozenBackbone-LearningCurve/18-FrozenBackbone-N100.yaml').read_text())
key = '+nmr_helpers.frozen_head_training.FrozenHeadOptimizer'
left, right = deepcopy(configs)
left['wandb']['name'] = right['wandb']['name']
left['fitting']['optimizer'][key]['scale_lr'] = right['fitting']['optimizer'][key]['scale_lr']
assert left == right, 'Pilots differ beyond name/scales LR'
for raw in configs:
    assert raw['data'] == base['data'] and raw['loss'] == base['loss']
    for field in ('trainer_kwargs', 'loader_kwargs', 'scheduler', 'early_stopping', 'keep', 'pre_fit_model', 'max_n_pre_fit'):
        assert raw['fitting'][field] == base['fitting'][field], field

# Native stopping decisions must survive interruption, including a nonzero wait.
early = EarlyStoppingWithLogging('valid/loss/total', patience=300)
early._evaluate_stopping_criteria(torch.tensor(.25))
for _ in range(117):
    assert not early._evaluate_stopping_criteria(torch.tensor(.3))[0]
early.state.update(total_checks=118, best_val_loss_check=1, best_val_loss=.25)
saved = deepcopy(early.state_dict())
restored = EarlyStoppingWithLogging('valid/loss/total', patience=300)
restored.load_state_dict(saved)
assert restored.wait_count == 117 and float(restored.best_score) == .25
for _ in range(183):
    assert early._evaluate_stopping_criteria(torch.tensor(.3))[0] == restored._evaluate_stopping_criteria(torch.tensor(.3))[0]
assert restored.wait_count == 300
legacy = EarlyStoppingWithLogging('valid/loss/total', patience=300)
legacy.load_state_dict({k: saved[k] for k in ('total_checks', 'best_val_loss_check', 'best_val_loss')})
assert legacy.wait_count == 117 and float(legacy.best_score) == .25

reports = []
initial_weights = None
original_factory = training.trainer_from_config


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


class Audit(Callback):
    def __init__(self, resume=None):
        self.resume = resume
        self.backbone_calls = 0

    def on_train_start(self, trainer, task):
        global initial_weights
        net = task.model.models['nequip']
        assert count_used_parameters(task.model, only_learnable=True) == 132
        self.before = {n: p.detach().clone() for n, p in task.model.named_parameters()}
        self.frozen_before = backbone_digest(net)
        assert len(trainer.train_dataloader.dataset) == 100
        valid_loader = trainer.val_dataloaders
        if isinstance(valid_loader, (list, tuple)):
            valid_loader = valid_loader[0]
        assert len(valid_loader.dataset) == 50
        self.cache = next(cb for cb in trainer.callbacks if isinstance(cb, FrozenFeatureCache)).report
        optimizer = trainer.optimizers[0]
        if self.resume:
            assert trainer.current_epoch == self.resume['epoch'] + 1
            assert trainer.global_step == self.resume['global_step']
            for group, old in zip(optimizer.param_groups, self.resume['optimizer_states'][0]['param_groups']):
                assert group['param_names'] == old['param_names']
                assert group['lr'] == old['lr']
                for parameter, number in zip(group['params'], old['params']):
                    for key, value in self.resume['optimizer_states'][0]['state'].get(number, {}).items():
                        assert torch.equal(optimizer.state[parameter][key].cpu(), value)
            assert trainer.lr_scheduler_configs[0].scheduler.state_dict() == self.resume['lr_schedulers'][0]
            cb = next(c for c in trainer.callbacks if isinstance(c, EarlyStoppingWithLogging))
            old = next(v for k, v in self.resume['callbacks'].items() if k.startswith('EarlyStopping'))
            assert cb.wait_count == old['wait_count'] and torch.equal(cb.best_score.cpu(), old['best_score'])
            assert all(r['hit'] for r in self.cache)
        else:
            if initial_weights is None:
                initial_weights = deepcopy(task.model.state_dict())
            else:
                for name, value in task.model.state_dict().items():
                    assert same(value, initial_weights[name]), name
            assert not optimizer.state
            groups = {g['name']: g for g in optimizer.param_groups}
            assert groups['normal']['lr'] == groups['non-decayable']['lr'] == .003
            assert groups['scales']['lr'] in (.003, .03)
            assert groups['normal']['weight_decay'] == .01
            assert groups['scales']['weight_decay'] == groups['non-decayable']['weight_decay'] == 0
            # Compare the cached model with the ORIGINAL TensorNequIP forward,
            # including gradients and one AdamW update (not just predictions).
            cached = deepcopy(task.model).train()
            full = deepcopy(task.model).train()
            full.models['nequip'].__class__ = TensorNequIP
            batch = to_batch([trainer.train_dataloader.dataset[i] for i in range(4)])
            pc, pf = cached(batch)['tensor'], full(batch)['tensor']
            torch.testing.assert_close(pc, pf, rtol=1e-5, atol=1e-4)
            lc = task.total_loss(cached, batch, {'tensor': pc}).loss_value
            lf = task.total_loss(full, batch, {'tensor': pf}).loss_value
            torch.testing.assert_close(lc, lf, rtol=1e-5, atol=1e-6)
            lc.backward(); lf.backward()
            for (name, p), (_, q) in zip(cached.named_parameters(), full.named_parameters()):
                if p.requires_grad:
                    torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5)
                else:
                    assert p.grad is None and q.grad is None
            # At baseline LR the new three-group optimizer must agree with
            # the launcher's original two-group AdamW update.
            new_opt = task.optimizer_factory(cached)
            old_opt = Optimizer('AdamW', lr=.003, weight_decay=.01)(full)
            new_opt.step(); old_opt.step()
            for (name, p), (_, q) in zip(cached.named_parameters(), full.named_parameters()):
                if groups['scales']['lr'] == .003 or 'scaler.per_element_scaling' not in name:
                    torch.testing.assert_close(p, q, rtol=1e-5, atol=2e-6)
                else:
                    assert not torch.equal(p, q)
        self.handle = net.layers[0].register_forward_pre_hook(self.count_call)

    def count_call(self, *args):
        self.backbone_calls += 1

    def on_train_epoch_start(self, trainer, task):
        if self.handle is None:
            self.handle = task.model.models['nequip'].layers[0].register_forward_pre_hook(self.count_call)

    def on_validation_epoch_end(self, trainer, task):
        # Remove test-only hooks before SaveBestModel serializes the model.
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def on_train_batch_start(self, trainer, task, batch, batch_idx):
        assert FEATURE_KEY in batch.other

    def on_train_end(self, trainer, task):
        assert self.backbone_calls == 0, self.backbone_calls
        if self.handle is not None:
            self.handle.remove()
        assert backbone_digest(task.model.models['nequip']) == self.frozen_before
        for name, parameter in task.model.named_parameters():
            if parameter.requires_grad:
                assert not torch.equal(parameter, self.before[name]), name
            else:
                assert torch.equal(parameter, self.before[name]), name
        assert trainer.global_step == 25 * trainer.current_epoch
        for split in ('train', 'valid'):
            assert torch.isfinite(trainer.callback_metrics[f'{split}/loss/total'])
            for element in ('Si', 'O'):
                assert torch.isfinite(trainer.callback_metrics[f'{split}/metrics/{element}_rmse'])
        self.result = dict(epochs=trainer.current_epoch, steps=trainer.global_step,
                           backbone_calls_during_training=self.backbone_calls, cache=self.cache,
                           valid_loss=float(trainer.callback_metrics['valid/loss/total']))


def make_factory(audit):
    def factory(config, output_dir):
        trainer = original_factory(config, output_dir)
        trainer.callbacks.append(audit)
        # Skip unrelated deployment only for the smoke run; model export and
        # real checkpoint callbacks still execute. Production keeps defaults.
        for callback in trainer.callbacks:
            if isinstance(callback, SaveBestModel):
                callback.try_to_deploy = False
        return trainer
    return factory


for path, config in zip(paths, configs):
    raw = deepcopy(config)
    raw['wandb'] = None
    raw['general'].update(root_dir=str(OUT), run_id=path.stem)
    raw['fitting']['trainer_kwargs'].update(accelerator='cpu', max_epochs=2)
    for split in ('train', 'valid'):
        raw['data'][split]['pre_transform'] = False
    raw['fitting']['callbacks'][0]['+nmr_helpers.frozen_head_training.FrozenFeatureCache']['directory'] = str(OUT / 'features')
    audit = Audit()
    training.trainer_from_config = make_factory(audit)
    try:
        train_from_config(raw)
    finally:
        training.trainer_from_config = original_factory
    run_dir = OUT / path.stem
    state = torch.load(run_dir / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
    assert state['epoch'] == 1 and state['global_step'] == 50
    smoke_config = yaml.safe_load((run_dir / 'train-config.yaml').read_text())
    smoke_config['fitting']['trainer_kwargs']['max_epochs'] = 3
    (run_dir / 'train-config.yaml').write_text(yaml.safe_dump(smoke_config))
    # Execute the same resume branch as train.sbatch, without the ARC/GPU setup.
    resume_audit = Audit(state)
    trainer_from_config = make_factory(resume_audit)
    old_argv = sys.argv
    sys.argv = ['train.sbatch', '--resume', str(run_dir)]
    try:
        start = launcher.index('    # Resume only a run CREATED WITH THIS SCRIPT.')
        branch = launcher[start:launcher.index('\nTRAIN_PY', start)]
        import textwrap
        exec(textwrap.dedent(branch), globals())
    finally:
        sys.argv = old_argv
    reports.append(dict(config=path.name, initial=audit.result, resumed=resume_audit.result,
                        state_restoration_verified=True))
    exported = torch.load(run_dir / 'model.pt', map_location='cpu', weights_only=False).eval()
    graph = data.valid[0]._replace(other={})
    original = deepcopy(exported)
    original.models['nequip'].__class__ = TensorNequIP
    with torch.no_grad():
        predicted = exported(graph)['tensor']
        torch.testing.assert_close(predicted, original(graph)['tensor'])
        torch.testing.assert_close(e3nn_script(exported)(graph)['tensor'], predicted)
    reports[-1]['export_without_cache_and_torchscript_verified'] = True
    print('PASS', path.name, reports[-1], flush=True)

# Exercise cache invalidation on a disposable model, never the saved runs.
model = torch.load(run_dir / 'model.pt', map_location='cpu', weights_only=False)
net = model.models['nequip']
dataset = GraphDataset([data.train[i]._replace(other={}) for i in range(2)])
directory = OUT / 'invalidation'
first = cache_dataset(dataset, net, directory)
assert not first['hit'] and cache_dataset(dataset, net, directory)['hit']
with torch.no_grad():
    net.tensor_readout.weight.add_(.01)
assert cache_dataset(dataset, net, directory)['path'] == first['path']
g = dataset.graphs[0]
positions = g.R.clone(); positions[0, 0] += .001
dataset.graphs[0] = g._replace(R=positions)
changed_graph = cache_dataset(dataset, net, directory)
assert not changed_graph['hit'] and changed_graph['path'] != first['path']
with torch.no_grad():
    next(net.layers[0].parameters()).add_(.001)
changed_backbone = cache_dataset(dataset, net, directory)
assert not changed_backbone['hit'] and changed_backbone['path'] != changed_graph['path']
next(net.layers[0].parameters()).requires_grad_(True)
try:
    cache_dataset(dataset, net, directory)
except ValueError:
    pass
else:
    raise AssertionError('Trainable backbone was accepted for caching')

(OUT / 'smoke-results.json').write_text(json.dumps(dict(pilots=reports,
    same_initial_weights=True, active_parameters=132, native_and_legacy_early_stopping=True,
    cached_predictions_gradients_and_updates_match=True, cache_invalidation_verified=True,
    original_config_settings_preserved=True, device='cpu'), indent=2))
print('Both full N100/50 pilots and exact launcher resume paths passed:', OUT, flush=True)
