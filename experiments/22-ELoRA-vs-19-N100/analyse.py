"""Compare selected checkpoints on validation only; no fitting or test access.

Run with the research project's .venv/bin/python. Outputs go beside this script.
"""
import os
from pathlib import Path
import sys
import json
import hashlib
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/elora-parity-matplotlib')
ROOT = Path(os.environ.get('NEQUIP_PROJECT', '/Users/louismcauliffe/code/research/NequIP_experiments'))
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from ase.io import read
from graph_pes.atomic_graph import AtomicGraph
import nmr_helpers
import nmr_helpers.elora

RUNS = {
    'Last layer + head': '19-LastLayerTransfer-LearningCurve/19-LastLayerTransfer-N100-9189909',
    'ELoRA rank 4 + head': '22-ELoRA-N100-R4-9201369',
}
COLORS = ['#198C83', '#7751B5']
torch.set_num_threads(4)
torch.set_default_dtype(torch.float32)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b

def evaluate():
    configs, records, models = {}, [], {}
    for label, folder in RUNS.items():
        directory = ROOT / 'graph-pes-results' / folder
        config = yaml.safe_load((directory / 'train-config.yaml').read_text())
        configs[label] = config
        assert config['data']['test'] is None
        assert config['data']['train']['n'] == 100
        assert config['data']['valid']['path'] == 'data/nmr_iso/valid.xyz'
        model = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
        ck = torch.load(directory / 'checkpoints/best.ckpt', map_location='cpu', weights_only=False)
        last = torch.load(directory / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
        weights = {k.removeprefix('model.'): v for k, v in ck['state_dict'].items() if k.startswith('model.')}
        exported = model.state_dict()
        matches = same(exported, weights)
        model.load_state_dict(weights, strict=True)
        models[label] = model.eval().requires_grad_(False)
        summary = yaml.safe_load((directory / 'summary.yaml').read_text())
        records.append(dict(method=label, run_id=directory.name,
            best_epoch=int(ck['epoch']) + 1, final_epoch=int(last['epoch']) + 1,
            best_step=int(ck['global_step']), final_step=int(last['global_step']),
            recorded_best_loss=summary['early_stopping/best_valid_loss'],
            checks_since_best=summary['early_stopping/checks_since_best'],
            runtime_minutes=summary['_runtime']/60,
            trainable_parameters=summary['n_learnable_parameters'],
            export_matches_best=matches,
            model_sha256=sha(directory/'model.pt'),
            best_checkpoint_sha256=sha(directory/'checkpoints/best.ckpt'),
            config_sha256=sha(directory/'train-config.yaml')))
    left, right = configs.values()
    for section in ('data', 'loss', 'fitting'):
        assert left[section] == right[section], f'Mismatched {section}'
    assert left['general']['seed'] == right['general']['seed'] == 42
    # Both wrappers enclose exactly the same source model and E/F checkpoint.
    assert next(iter(left['model'].values()))['model'] == next(iter(right['model'].values()))['model']
    frames = read(ROOT/'data/nmr_iso/valid.xyz', index=':')
    assert len(frames) == 50
    rows = []
    start = time.perf_counter()
    for i, atoms in enumerate(frames):
        target = np.asarray(atoms.arrays['MS_Iso_QM'], dtype=np.float64).reshape(-1)
        np.testing.assert_allclose(target, np.trace(atoms.arrays['ms'].reshape(-1,3,3), axis1=1, axis2=2)/3, atol=1e-6, rtol=0)
        assert (atoms.numbers == 14).sum() == 48 and (atoms.numbers == 8).sum() == 96
        graph = AtomicGraph.from_ase(atoms, cutoff=6.)
        assert not graph.properties and not graph.other
        with torch.inference_mode():
            for label, model in models.items():
                pred = model.predict(graph, ['tensor'])['tensor'].cpu().numpy().astype(np.float64).reshape(-1)
                assert pred.shape == target.shape and np.isfinite(pred).all()
                rows.append(pd.DataFrame(dict(method=label, structure=i,
                    atom=np.arange(len(atoms)), element=atoms.get_chemical_symbols(),
                    reference_ppm=target, prediction_ppm=pred, error_ppm=pred-target)))
        if (i+1) % 10 == 0:
            print(f'Predicted both models: {i+1}/50 structures in {time.perf_counter()-start:.1f}s', flush=True)
    sites = pd.concat(rows, ignore_index=True)
    metrics = []
    for (method, element), group in sites.groupby(['method', 'element'], sort=False):
        err = group.error_ppm.to_numpy()
        target = group.reference_ppm.to_numpy()
        metrics.append(dict(method=method, element=element, sites=len(group),
            rmse_ppm=float(np.sqrt(np.mean(err**2))), mae_ppm=float(np.mean(abs(err))),
            bias_ppm=float(np.mean(err)), r2=float(1-np.sum(err**2)/np.sum((target-target.mean())**2))))
    metrics = pd.DataFrame(metrics)
    loss_params = next(iter(left['loss']['isotropy'].values()))
    structure_rows = []
    for (method, structure), group in sites.groupby(['method', 'structure']):
        mse = group.groupby('element').error_ppm.apply(lambda x: np.mean(x**2))
        loss = .5*(mse['Si']/loss_params['si_std']**2 + mse['O']/loss_params['o_std']**2)
        structure_rows.append(dict(method=method, structure=structure, balanced_mse=loss))
    structures = pd.DataFrame(structure_rows)
    for record in records:
        loss = float(structures[structures.method == record['method']].balanced_mse.mean())
        record['evaluated_balanced_mse'] = loss
        np.testing.assert_allclose(loss, record['recorded_best_loss'], rtol=5e-4, atol=1e-7)
    sites.to_csv(OUT/'validation-sites.csv.gz', index=False)
    metrics.to_csv(OUT/'validation-metrics.csv', index=False)
    structures.to_csv(OUT/'validation-structures.csv', index=False)
    pd.DataFrame(records).to_csv(OUT/'runs.csv', index=False)
    manifest = dict(runs=records, split='validation', training_structures=100,
        validation_structures=50, validation_sites=7200, seed=42,
        matched_sections=['data', 'loss', 'fitting', 'source model and E/F checkpoint'],
        target='Absolute DFT isotropic shielding, MS_Iso_QM = trace(ms)/3, ppm',
        validation_sha256=sha(ROOT/'data/nmr_iso/valid.xyz'),
        train_file_sha256=sha(ROOT/'data/nmr_iso/train.xyz'),
        source_EF_sha256=sha(ROOT/'graph-pes-results/Kavanagh-Med-9059603/model.pt'),
        elora_implementation_sha256=sha(ROOT/'nmr_helpers/elora.py'),
        evaluator_sha256=sha(__file__), test_accessed=False,
        inference='CPU float32, 4 threads; metrics float64; strict best-checkpoint load',
        torch=torch.__version__, numpy=np.__version__)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(metrics.to_string(index=False), flush=True)
    print(pd.DataFrame(records)[['method','best_epoch','final_epoch','evaluated_balanced_mse','export_matches_best']].to_string(index=False), flush=True)
    return sites, metrics, structures

def plot(sites, metrics, structures):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    labels = list(RUNS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for row, element in enumerate(['Si', 'O']):
        subset = sites[sites.element == element]
        low = min(subset.reference_ppm.min(), subset.prediction_ppm.min())
        high = max(subset.reference_ppm.max(), subset.prediction_ppm.max())
        margin = .06*(high-low)
        limits = [low-margin, high+margin]
        for col, method in enumerate(labels):
            ax = axes[row, col]
            data = subset[subset.method == method]
            m = metrics[(metrics.method == method) & (metrics.element == element)].iloc[0]
            ax.scatter(data.reference_ppm, data.prediction_ppm, s=9, alpha=.24,
                c=COLORS[col], edgecolors='none', rasterized=True)
            ax.plot(limits, limits, '--', c='#414956', lw=1, zorder=0)
            ax.set(xlim=limits, ylim=limits, aspect='equal',
                xlabel='DFT isotropic shielding (ppm)', ylabel='Predicted isotropic shielding (ppm)')
            ax.set_title(f'{element} · {method}', loc='left', weight='bold', fontsize=12)
            ax.text(.04, .96, f'RMSE  {m.rmse_ppm:.3f} ppm\nMAE    {m.mae_ppm:.3f} ppm\n$R^2$      {m.r2:.4f}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=4))
            ax.grid(alpha=.12)
    fig.suptitle('ELoRA vs last-layer transfer · N = 100', fontsize=19, weight='bold', y=.985)
    fig.text(.5, .945, 'Best validation checkpoints · identical 50 validation structures · seed 42', ha='center', color='#535D6B')
    fig.text(.5, .018, '2,400 Si sites and 4,800 O sites per model · matched axes within each element · dashed line: perfect prediction', ha='center', fontsize=10, color='#535D6B')
    fig.subplots_adjust(top=.89, bottom=.09, left=.09, right=.97, hspace=.30, wspace=.27)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'validation-parity.{ext}', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, metric, title in zip(axes[:2], ['rmse_ppm','mae_ppm'], ['RMSE (ppm)', 'MAE (ppm)']):
        for i, method in enumerate(labels):
            vals = [float(metrics[(metrics.method==method)&(metrics.element==e)][metric].iloc[0]) for e in ['Si','O']]
            bars = ax.bar(np.arange(2)+(i-.5)*.35, vals, width=.33, color=COLORS[i], label=method)
            ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=10)
        ax.set_xticks([0,1], ['Si','O'])
        ax.set(title=title, ylim=(0, ax.get_ylim()[1]*1.16))
        ax.grid(axis='y', alpha=.15)
    paired = structures.pivot(index='structure', columns='method', values='balanced_mse')
    wins = int((paired[labels[1]] < paired[labels[0]]).sum())
    max_loss = float(paired.to_numpy().max())*1.06
    ax = axes[2]
    ax.scatter(paired[labels[0]], paired[labels[1]], color=COLORS[1], s=28, alpha=.75)
    ax.plot([0,max_loss], [0,max_loss], '--', color='#535D6B', lw=1)
    ax.set(xlim=(0,max_loss), ylim=(0,max_loss), xlabel='Last layer + head', ylabel='ELoRA rank 4 + head', title='Balanced MSE per structure')
    ax.text(.04,.96,f'ELoRA lower on {wins}/50 structures', transform=ax.transAxes, va='top', fontsize=10)
    fig.suptitle('Matched validation errors · N = 100', fontsize=17, weight='bold', y=.99)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels, loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(top=.83,bottom=.19,left=.06,right=.98,wspace=.30)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'validation-comparison.{ext}', dpi=200)
    plt.close(fig)
    print(f'ELoRA has lower balanced loss on {wins}/50 validation structures.', flush=True)

if __name__ == '__main__':
    plot(*evaluate())
