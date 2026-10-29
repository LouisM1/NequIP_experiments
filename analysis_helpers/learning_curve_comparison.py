"""Reproduce the five-method scalar NMR comparison from audited predictions."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, FixedLocator, LogLocator

ROOT=Path('/Users/louismcauliffe/code/research/NequIP_experiments')
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,default=ROOT/'experiments/21-23-LearningCurve-data')
parser.add_argument('--output',type=Path,default=ROOT/'experiments/21-23-LearningCurve-figures')
args=parser.parse_args()
DATA,OUT=args.data,args.output
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA/'metrics.csv')
manifest=json.loads((DATA/'manifest.json').read_text())
N=[10,50,100,200,800]
ORDER=[23,22,19,21,17]
LABEL={23:'Full fine-tuning',22:'ELoRA',19:'Last layer + NMR head',21:'NMR head only',17:'Direct training'}
COLOR={23:'#7548A3',22:'#008C83',19:'#377BB5',21:'#7D858D',17:'#D18B35'}
MARKER={23:'D',22:'o',19:'s',21:'v',17:'^'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,
    'axes.titleweight':'semibold','axes.labelcolor':'#333E4D','text.color':'#253143',
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BCC2C9',
    'xtick.color':'#566171','ytick.color':'#566171','axes.labelpad':9,
    'savefig.facecolor':'white','figure.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
paths=[]
def save(fig,name):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=260,bbox_inches='tight')
    paths.append(name)
    plt.close(fig)
def size_axes(ax,metric):
    ax.set(xscale='log',yscale='log',xlim=(8,1000),xlabel='Training structures, N')
    ax.xaxis.set_major_locator(FixedLocator(N)); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y',which='major',color='#E4E8ED',lw=.8)
    ax.grid(axis='y',which='minor',color='#F0F2F5',lw=.5)
    ax.set_axisbelow(True)
    ax.set_ylabel('Element-balanced standardised MSE' if metric=='loss' else 'RMSE (ppm)')
    ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
def draw(ax,g,exp,metric,style='-',label=None):
    g=g.sort_values('n')
    assert g.n.tolist()==N
    assert np.isfinite(g[metric]).all() and (g[metric]>0).all()
    ax.plot(g.n,g[metric],color=COLOR[exp],lw=2.1,ls=style,marker=MARKER[exp],
            ms=6.2,mec='white',mew=.7,label=label or LABEL[exp],zorder=4 if exp in (22,23) else 3)
    provisional=g[~g.status.isin(['Early stopping', 'Converged (researcher assessment)'])]
    if not provisional.empty:
        ax.scatter(provisional.n,provisional[metric],s=48,marker=MARKER[exp],facecolor='white',
                   edgecolor=COLOR[exp],lw=1.8,zorder=6)
def footer(fig,text):
    fig.text(.065,.02,text,fontsize=9,color='#687383',va='bottom')

for split,split_label in [('test','Held-out test'),('valid','Validation')]:
    fig,axes=plt.subplots(1,3,figsize=(15.3,5.8))
    for i,(ax,metric,title) in enumerate(zip(axes,['Si_rmse','O_rmse','loss'],['Silicon','Oxygen','Balanced error'])):
        for exp in ORDER:
            g=df.query('experiment==@exp and primary and split==@split')
            draw(ax,g,exp,metric)
        size_axes(ax,metric)
        ax.set_title(title,loc='left',pad=15)
    fig.suptitle('How much NMR data does each method need?',x=.065,y=.99,ha='left',fontsize=21,weight='bold')
    fig.text(.065,.915,f'{split_label} performance · logarithmic axes · lower is better',fontsize=11,color='#687383')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='upper left',bbox_to_anchor=(.055,.88),frameon=False,columnspacing=1.7,fontsize=10)
    footer(fig,'Best validation checkpoints · 150 test / 50 validation structures · seed 42 · one sweep per method.\n'
               'Lines join measured sizes; one seed, no seed error bars. NMR head only = Experiment 21.')
    fig.subplots_adjust(left=.065,right=.985,bottom=.23,top=.74,wspace=.32)
    save(fig,f'comparison-{split}-log-log')

for exp in ORDER:
    fig,axes=plt.subplots(1,3,figsize=(13.8,5.4))
    for ax,metric,title in zip(axes,['Si_rmse','O_rmse','loss'],['Silicon','Oxygen','Balanced error']):
        for split,style,label in [('test','-','Held-out test'),('valid','--','Validation')]:
            draw(ax,df.query('experiment==@exp and primary and split==@split'),exp,metric,style,label)
        size_axes(ax,metric); ax.set_title(title,loc='left',pad=14)
    axes[0].legend(frameon=False,fontsize=10)
    fig.suptitle(f'Experiment {exp} · {LABEL[exp]}',x=.07,y=.98,ha='left',fontsize=20,weight='bold')
    fig.text(.07,.89,'Prediction error versus training-set size · logarithmic axes',fontsize=11,color='#687383')
    footer(fig,'Same validation-selected checkpoints on both splits · seed 42 · one run per size.\n'+
               'Targets: absolute isotropic NMR shielding (ppm). N excludes the 50 validation structures.')
    fig.subplots_adjust(left=.07,right=.985,bottom=.24,top=.77,wspace=.34)
    save(fig,f'{exp}-learning-curve-log-log')

for split in ('valid','test'):
    predictions=dict(np.load(DATA/f'{split}-predictions.npz'))
    y=predictions['reference']; z=predictions['atomic_numbers']
    # Common reference-led limits per element, expanded to contain every displayed prediction.
    limits={}
    for number,element in [(14,'Si'),(8,'O')]:
        mask=z==number
        vals=[y[mask]]+[predictions[r.run_id][mask] for r in df.query('primary and split==@split').itertuples()]
        lo=min(v.min() for v in vals); hi=max(v.max() for v in vals); pad=.055*(hi-lo)
        limits[element]=(lo-pad,hi+pad)
    def parity(ax,row,element,number):
        mask=z==number; pr=predictions[row.run_id][mask]; ref=y[mask]
        ax.scatter(ref,pr,s=2.3,c=COLOR[int(row.experiment)],alpha=.17,edgecolors='none',rasterized=True)
        lo,hi=limits[element]
        ax.plot([lo,hi],[lo,hi],ls='--',color='#434C58',lw=1,zorder=4)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect='equal')
        ax.text(.04,.96,f'RMSE {getattr(row,element+"_rmse"):.2f} ppm\n'+r'$R^2$'+f' {getattr(row,element+"_r2"):.3f}',
                transform=ax.transAxes,va='top',fontsize=9,
                bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
        ax.tick_params(labelsize=8)
    for exp in ORDER:
        group=df.query('experiment==@exp and primary and split==@split').sort_values('n')
        fig,axes=plt.subplots(2,5,figsize=(16.5,7.3))
        for col,row in enumerate(group.itertuples()):
            for i,(number,element) in enumerate([(14,'Si'),(8,'O')]):
                ax=axes[i,col]; parity(ax,row,element,number)
                if i==0: ax.set_title(f'N = {row.n}'+(' *' if row.status not in ('Early stopping', 'Converged (researcher assessment)') else ''),pad=12)
                if i==1: ax.set_xlabel('DFT shielding (ppm)',fontsize=9)
                if col==0: ax.set_ylabel(f'{element} · predicted shielding (ppm)',fontsize=10)
        fig.suptitle(f'Experiment {exp} · {LABEL[exp]}',x=.065,y=.995,ha='left',fontsize=20,weight='bold')
        fig.text(.065,.935,('Held-out test' if split=='test' else 'Validation')+' parity · all sites · common element-wise limits across methods and N',fontsize=11,color='#687383')
        footer(fig,'Dashed line: ideal agreement. Best validation checkpoints; absolute isotropic shieldings, not referenced chemical shifts.\n'
                   f'{len(np.unique(predictions["structure"]))} structures · {(z==14).sum():,} silicon sites · {(z==8).sum():,} oxygen sites · seed 42.')
        fig.subplots_adjust(left=.065,right=.99,bottom=.16,top=.88,wspace=.28,hspace=.24)
        save(fig,f'{exp}-{split}-parity')
    fig,axes=plt.subplots(2,5,figsize=(16.5,7.3))
    for col,exp in enumerate(ORDER):
        row=next(df.query('experiment==@exp and primary and split==@split and n==800').itertuples())
        for i,(number,element) in enumerate([(14,'Si'),(8,'O')]):
            ax=axes[i,col];parity(ax,row,element,number)
            if i==0:ax.set_title(LABEL[exp]+(' *' if row.status not in ('Early stopping', 'Converged (researcher assessment)') else ''),fontsize=11,pad=12)
            if i==1:ax.set_xlabel('DFT shielding (ppm)',fontsize=9)
            if col==0:ax.set_ylabel(f'{element} · predicted shielding (ppm)',fontsize=10)
    fig.suptitle('Five approaches · the same 800 training structures',x=.065,y=.995,ha='left',fontsize=20,weight='bold')
    fig.text(.065,.935,('Held-out test' if split=='test' else 'Validation')+' parity · identical evaluation sites · common limits across methods',fontsize=11,color='#687383')
    footer(fig,'Dashed line: ideal agreement. Checkpoints selected only on validation loss.\nAbsolute isotropic shieldings in ppm · seed 42 · one trained run per method.')
    fig.subplots_adjust(left=.065,right=.99,bottom=.16,top=.88,wspace=.28,hspace=.24)
    save(fig,f'comparison-{split}-parity-N800')

# Keep the duplicate ELoRA N=100 result visible without choosing a run by test score.
df.query('experiment==22 and n==100').to_csv(OUT/'elora-N100-pilot-and-sweep.csv',index=False)
df.query('primary').to_csv(OUT/'plotted-metrics.csv',index=False)
(OUT/'figure-index.json').write_text(json.dumps({'figures':paths,'formats':['png','pdf','svg'],'axis_scales':'log-log',
    'comparison_methods':LABEL,'primary_split':'test','uncertainty':'No seed bands: one seed',
    'source_manifest':str(DATA/'manifest.json')},indent=2)+'\n')
print('Saved',len(paths),'figures in PNG, PDF and SVG:',OUT)
