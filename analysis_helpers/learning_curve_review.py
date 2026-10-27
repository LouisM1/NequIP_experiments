"""Replot experiments 17–19 from the verified, saved epoch and checkpoint tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator

METHODS = {17: 'Scratch', 18: 'Frozen backbone', 19: 'Last layer + head'}
COLORS = {17: '#386CB0', 18: '#D98535', 19: '#198C83'}
TRAIN, VALID, BEST = '#21918C', '#354A78', '#C79520'
DATA_NAME = '17-19-LearningCurve-data'
FIGURE_NAME = '17-19-LearningCurve-figures'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'semibold', 'savefig.facecolor': 'white', 'pdf.fonttype': 42})

def load_tables(root):
    directory = Path(root) / 'experiments' / DATA_NAME
    return pd.read_csv(directory / 'runs.csv'), pd.read_csv(directory / 'epoch-loss.csv')

def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return out / f'{name}.png'

def _epoch_panel(ax, row, epochs, legend=False):
    history = epochs[epochs.run_id == row.run_id].sort_values('epoch')
    ax.plot(history.epoch, history.train_loss, color=TRAIN, lw=1.2, label='Training — epoch mean')
    ax.plot(history.epoch, history.valid_loss, color=VALID, lw=1.5, label='Validation')
    ax.axvline(row.best_epoch, color=BEST, linestyle=':', lw=1.2)
    ax.scatter([row.best_epoch], [row.best_valid_loss], color=BEST, edgecolor='white',
               s=46, linewidth=.7, zorder=5, label='Selected checkpoint')
    ax.set(xlabel='Epoch', ylabel='Balanced standardised MSE', yscale='log',
           xlim=(1, max(row.final_epoch * 1.025, 2)))
    ax.grid(axis='y', alpha=.18)
    ax.set_title(f'N = {row.n:,}  ·  best epoch {row.best_epoch:,}', loc='left', fontsize=12)
    if legend:
        ax.legend(frameon=False, fontsize=9, loc='best')

def _size_axes(ax):
    sizes = [10, 50, 100, 200, 800]
    ax.set(xscale='log', yscale='log', xlabel='Training structures, N',
           ylabel='Best validation loss', xlim=(8, 1000))
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(which='major', alpha=.18)

def _size_line(ax, group, method, label=None):
    group = group.sort_values('n')
    ax.plot(group.n, group.best_valid_loss, color=COLORS[method], lw=2,
            linestyle='--' if method == 18 else '-', label=label or METHODS[method])
    for provisional, points in group.groupby('provisional'):
        ax.scatter(points.n, points.best_valid_loss, s=62, marker='o',
                   facecolor='white' if provisional else COLORS[method],
                   edgecolor=COLORS[method], linewidth=1.7, zorder=4)

def plot_run(runs, epochs, run_id, out):
    row = runs.set_index('run_id').loc[run_id].copy()
    row['run_id'] = run_id
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _epoch_panel(ax, row, epochs, legend=True)
    fig.suptitle(f'Experiment {row.experiment} · {row.method}', x=.105, ha='left', fontsize=17, weight='bold')
    fig.text(.105, .015, f'{row.status} · {row.final_epoch:,} completed epochs · '
             f'best validation loss {row.best_valid_loss:.5g}', fontsize=10, color='#49515C')
    fig.subplots_adjust(top=.82, bottom=.15, left=.105, right=.98)
    return _save(fig, out, run_id + '-training-curve')

def plot_method(runs, epochs, experiment, out):
    group = runs[runs.experiment == experiment].sort_values('n')
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for ax, (_, row) in zip(axes.flat, group.iterrows()):
        _epoch_panel(ax, row, epochs, legend=False)
        ax.text(.98, .97, row.status, ha='right', va='top', transform=ax.transAxes,
                fontsize=9, color='#59616B', bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
    ax = axes.flat[-1]
    _size_line(ax, group, experiment)
    _size_axes(ax)
    ax.set_title('Best validation loss vs training-set size', loc='left', fontsize=12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .955), ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f'Experiment {experiment} · {METHODS[experiment]} learning curves', fontsize=19, weight='bold', y=.985)
    footnote = ('Open markers: frozen runs have not reached early stopping.' if experiment == 18
                else 'Selected checkpoints minimise loss on the same 50 validation structures.')
    fig.text(.07, .015, footnote + '  Seed 42 · unsmoothed epoch histories.', fontsize=10, color='#59616B')
    fig.subplots_adjust(top=.90, bottom=.07, left=.08, right=.98, hspace=.40, wspace=.26)
    return _save(fig, out, f'{experiment}-learning-curves')

def plot_comparison(runs, out):
    # Transform once, then use linear axes to show the logarithms explicitly.
    values = runs[['n', 'best_valid_loss']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Log-log comparison requires finite, positive sizes and losses.')
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for method in (17, 18, 19):
        group = runs[runs.experiment == method].sort_values('n')
        ax.plot(np.log10(group.n), np.log10(group.best_valid_loss),
                color=COLORS[method], lw=2,
                linestyle='--' if method == 18 else '-', label=METHODS[method])
        for provisional, points in group.groupby('provisional'):
            ax.scatter(np.log10(points.n), np.log10(points.best_valid_loss),
                       s=62, marker='o', facecolor='white' if provisional else COLORS[method],
                       edgecolor=COLORS[method], linewidth=1.7, zorder=4)
    sizes = np.array([10, 50, 100, 200, 800])
    ax.set(xlabel=r'$\log_{10}(N)$ — training structures',
           ylabel=r'$\log_{10}(\mathrm{best\ validation\ loss})$',
           xlim=np.log10([8, 1000]))
    ax.set_xticks(np.log10(sizes), [f'{np.log10(n):.2f}' for n in sizes])
    ax.grid(which='major', alpha=.18)
    ax.legend(frameon=False, loc='lower left', fontsize=11)
    fig.suptitle('NMR data efficiency · log–log', x=.11, ha='left', fontsize=21, weight='bold')
    ax.set_title('Best checkpoint loss · same 50 validation structures · seed 42',
                 loc='left', fontsize=11, weight='normal', pad=15)
    fig.text(.11, .035, 'Open markers / dashed line: frozen runs are provisional; none reached early stopping.\n'
             'Base-10 logarithms; N = 10, 50, 100, 200, 800. Lines join measured sizes; one seed.',
             fontsize=9.5, color='#59616B')
    fig.subplots_adjust(top=.83, bottom=.20, left=.11, right=.98)
    return _save(fig, out, '17-19-data-efficiency')

def plot_rmse(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, element in zip(axes, ('Si', 'O')):
        for method in (17, 18, 19):
            group = runs[runs.experiment == method].sort_values('n')
            ax.plot(group.n, group[element + '_rmse'], color=COLORS[method], lw=2,
                    linestyle='--' if method == 18 else '-', marker='o',
                    markerfacecolor='white' if method == 18 else COLORS[method],
                    label=METHODS[method])
        _size_axes(ax)
        ax.set(title=element, ylabel='Validation RMSE (ppm)')
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle('Element-wise accuracy at the selected checkpoints', fontsize=17, weight='bold')
    fig.text(.07, .025, 'Fresh evaluation on the same validation sites. Frozen curves remain provisional.', fontsize=10, color='#59616B')
    fig.subplots_adjust(top=.82, bottom=.18, left=.07, right=.98, wspace=.23)
    return _save(fig, out, '17-19-validation-rmse')
