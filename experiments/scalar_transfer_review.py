"""Shared plotting and verified-cache reader for the three lean notebooks."""
from pathlib import Path
import hashlib,json,subprocess,sys,os
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
COLORS={'09':'#137C72','10':'#C53F78','11':'#8B969E','12':'#4279D8','mean':'#3D454D'}
LABELS={'09':'09  Full fine-tuning','10':'10  Scratch, LR 1e−4','11':'11  Frozen, LR 1e−4','12':'12  Frozen, LR 3e−3','mean':'Training-mean baseline'}
plt.rcParams.update({'figure.dpi':130,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#36424A','text.color':'#26323B','axes.edgecolor':'#B7BFC5','xtick.color':'#59636B','ytick.color':'#59636B','legend.frameon':False,'savefig.facecolor':'white'})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_review(project,cache):
 project,cache=Path(project),Path(cache)
 manifest_path=cache/'manifest.json'
 if not manifest_path.exists():
  env=dict(os.environ,NEQUIP_PROJECT=str(project),NEQUIP_EVAL_OUTPUT=str(cache))
  subprocess.run([sys.executable,str(Path(__file__).with_name('scalar_transfer_evaluate.py'))],env=env,check=True)
 manifest=json.loads(manifest_path.read_text());prov=pd.read_csv(cache/'provenance.csv',dtype={'run':str})
 for split,digest in manifest['data_sha256'].items():assert sha(project/f'data/nmr_iso/{split}.xyz')==digest,f'{split} data changed; regenerate cache.'
 for r in prov.to_dict('records'):
  folder=project/'graph-pes-results'/r['directory']
  for f,key in [('model.pt','model_sha256'),('checkpoints/best.ckpt','checkpoint_sha256'),('checkpoints/last.ckpt','last_checkpoint_sha256'),('train-config.yaml','config_sha256')]:
   assert sha(folder/f)==r[key],f'{r["run"]}: {f} changed; regenerate cache.'
 if 'prediction_cache_sha256' in manifest:assert sha(cache/'sites.csv.gz')==manifest['prediction_cache_sha256'],'Prediction cache changed; regenerate.'
 sites=pd.read_csv(cache/'sites.csv.gz',dtype={'run':str})
 ref=sites[sites.run=='09'].copy()
 means=ref[ref.split=='train'].groupby('element').reference.mean()
 ref['prediction']=ref.element.map(means);ref['error']=ref.prediction-ref.reference;ref['run']='mean'
 sites=pd.concat([sites,ref],ignore_index=True)
 metrics=sites.groupby(['run','split','element'],sort=False).error.agg(N='size',MAE=lambda x:np.mean(np.abs(x)),RMSE=lambda x:np.sqrt(np.mean(x*x)),bias='mean').reset_index()
 structures=sites.assign(SE=sites.error**2,AE=np.abs(sites.error)).groupby(['run','split','element','structure']).agg(N=('error','size'),SSE=('SE','sum'),SAE=('AE','sum')).reset_index()
 structures['RMSE']=np.sqrt(structures.SSE/structures.N)
 rng=np.random.default_rng(42);draws=rng.integers(0,50,size=(4000,50));boot={}
 for (run,el),g in structures[structures.split=='valid'].groupby(['run','element']):
  g=g.sort_values('structure');n=g.N.to_numpy()[draws].sum(axis=1)
  boot[run,el,'RMSE']=np.sqrt(g.SSE.to_numpy()[draws].sum(axis=1)/n)
  boot[run,el,'MAE']=g.SAE.to_numpy()[draws].sum(axis=1)/n
 return sites,metrics,structures,boot,prov,manifest

def value(metrics,run,el,metric='RMSE',split='valid'):
 return float(metrics.query('run==@run and element==@el and split==@split')[metric].iloc[0])
def save(fig,out,name):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.show();plt.close(fig)
def overview(prov,runs):
 cols=['run','best_epoch','initial_lr','trainable_parameters','export_matches_checkpoint']
 display(prov[prov.run.isin(runs)][cols].rename(columns={'run':'Experiment','best_epoch':'Selected epoch','initial_lr':'Initial LR','trainable_parameters':'Trainable parameters','export_matches_checkpoint':'Export matches checkpoint'}).set_index('Experiment'))

def paired_figure(metrics,boot,base,new,title,out,name):
 fig,axes=plt.subplots(2,2,figsize=(11.5,7.4),layout='constrained')
 rows=[]
 for col,el in enumerate(['Si','O']):
  for row,metric in enumerate(['MAE','RMSE']):
   ax=axes[row,col]
   for j,run in enumerate([base,new]):
    v=value(metrics,run,el,metric);lo,hi=np.quantile(boot[run,el,metric],[.025,.975])
    ax.plot([lo,hi],[j,j],color=COLORS[run],lw=3);ax.scatter(v,j,color=COLORS[run],s=70,zorder=3)
    ax.annotate(f'{v:.3f}',(hi,j),xytext=(7,0),textcoords='offset points',va='center',weight='bold')
   old=value(metrics,base,el,metric);v=value(metrics,new,el,metric);delta=boot[new,el,metric]-boot[base,el,metric];lo,hi=np.quantile(delta,[.025,.975]);pct=100*(old-v)/old
   rows.append(dict(element=el,metric=metric,baseline=base,comparison=new,old=old,new=v,reduction_pct=pct,delta=v-old,lower=lo,upper=hi))
   upper=max(np.quantile(boot[r,el,metric],.975) for r in [base,new])
   ax.set(xlim=(0,upper*1.4),ylim=(1.7,-.65),yticks=[0,1],yticklabels=[LABELS[r] for r in [base,new]],xlabel=f'{metric} (ppm)',title=f'{el} · {metric}')
   ax.text(.02,.03,f'{pct:+.1f}% reduction\nPaired Δ {v-old:+.3f} [{lo:+.3f}, {hi:+.3f}] ppm',transform=ax.transAxes,fontsize=10,va='bottom',bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
   ax.grid(axis='x',alpha=.15)
 fig.suptitle(title,fontsize=17,weight='bold');save(fig,out,name)
 result=pd.DataFrame(rows);result.to_csv(Path(out)/(name+'-paired-changes.csv'),index=False);return result

def gap_figure(metrics,runs,title,out,name,baseline=True):
 fig,axes=plt.subplots(1,2,figsize=(11.5,5.6),layout='constrained')
 for ax,el in zip(axes,['Si','O']):
  for j,run in enumerate(runs):
   vals=[value(metrics,run,el,split=s) for s in ['train','valid']]
   x=np.array([0,1])+(j-(len(runs)-1)/2)*.045
   ax.plot(x,vals,'o-',lw=2.3,ms=7,color=COLORS[run],label=LABELS[run])
   # Values are tabulated below when several small errors overlap.
   if run in ['11','12']:
    for xx,y in zip(x,vals):ax.annotate(f'{y:.2f}',(xx,y),xytext=(7,-16 if run=='12' and el=='Si' else 7),textcoords='offset points',fontsize=10,color=COLORS[run])
  if baseline:
   b=value(metrics,'mean',el);ax.axhline(b,color='#46505A',ls='--',lw=1.2,label=f'Mean baseline · validation {b:.2f}')
  ax.set(xticks=[0,1],xticklabels=['Training · 800 structures','Validation · 50 structures'],xlim=(-.18,1.4),ylim=(0,None),ylabel='RMSE (ppm)',title=el)
  ax.grid(axis='y',alpha=.15);ax.legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.12))
 fig.suptitle(title,fontsize=17,weight='bold');save(fig,out,name)
 table=metrics[metrics.run.isin(runs)].pivot(index=['run','element'],columns='split',values='RMSE').rename(columns={'train':'Training RMSE','valid':'Validation RMSE'})
 display(table.round(3));table.to_csv(Path(out)/(name+'-values.csv'))

def structure_figure(structures,base,new,title,out,name):
 fig,axes=plt.subplots(1,2,figsize=(11.4,5.2),layout='constrained');rows=[]
 for ax,el in zip(axes,['Si','O']):
  g=structures.query('split=="valid" and element==@el');p=g.pivot(index='structure',columns='run',values='RMSE')
  improves=p[new]<p[base];limit=max(p[base].max(),p[new].max())*1.13
  ax.scatter(p[base],p[new],s=40,c=np.where(improves,COLORS[new],'#C88942'),alpha=.8,edgecolors='white',linewidths=.4)
  ax.plot([0,limit],[0,limit],'--',color='#727C83',lw=1)
  for idx in p[[base,new]].max(axis=1).nlargest(2).index:
   ax.annotate(str(idx),(p.loc[idx,base],p.loc[idx,new]),xytext=(6,7),textcoords='offset points',fontsize=9)
  ax.set(xlim=(0,limit),ylim=(0,limit),aspect='equal',xlabel=f'{LABELS[base]} · RMSE (ppm)',ylabel=f'{LABELS[new]} · RMSE (ppm)',title=f'{el} · improves {int(improves.sum())}/50 structures')
  ax.text(.04,.95,'Below diagonal = improvement',transform=ax.transAxes,va='top',fontsize=10);ax.grid(alpha=.12)
  ss=g.pivot(index='structure',columns='run',values='SSE');gain=ss[base]-ss[new];net=gain.sum();largest=gain.idxmax()
  rows.append(dict(element=el,structures_improved=int(improves.sum()),largest_contributor=int(largest),share_of_net_SSE_reduction_pct=100*gain.loc[largest]/net if net>0 else np.nan))
 fig.suptitle(title,fontsize=17,weight='bold');save(fig,out,name)
 result=pd.DataFrame(rows);result.to_csv(Path(out)/(name+'-concentration.csv'),index=False);return result

def parity_figure(sites,run,title,out,name):
 fig,axes=plt.subplots(1,2,figsize=(11.5,5.2),layout='constrained')
 for ax,el in zip(axes,['Si','O']):
  g=sites.query('split=="valid" and run==@run and element==@el');low=min(g.reference.min(),g.prediction.min());high=max(g.reference.max(),g.prediction.max());pad=(high-low)*.08
  ax.scatter(g.reference,g.prediction,color=COLORS[run],s=10,alpha=.3,edgecolors='none',rasterized=True)
  ax.plot([low-pad,high+pad],[low-pad,high+pad],'--',color='#515B64',lw=1.3)
  ax.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),aspect='equal',xlabel='DFT shielding (ppm)',ylabel='Predicted shielding (ppm)',title=f'{el} · all {len(g):,} validation sites')
  ax.text(.04,.95,f'MAE {g.error.abs().mean():.2f} ppm\nRMSE {np.sqrt(np.mean(g.error**2)):.2f} ppm\nBias {g.error.mean():+.2f} ppm',transform=ax.transAxes,va='top',bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
 fig.suptitle(title,fontsize=17,weight='bold');save(fig,out,name)
