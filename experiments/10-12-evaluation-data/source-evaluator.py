from pathlib import Path
import sys,os,json,hashlib,time,copy
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np,pandas as pd,torch,yaml
from ase.io import read
from graph_pes.atomic_graph import AtomicGraph
ROOT=Path('/Users/louismcauliffe/code/research/NequIP_experiments')
OUT=Path('/Users/louismcauliffe/.codex/.chatgpt-projects/g-p-6aa93f0341cc8191bbd47fb85a4101fb/outputs/experiments-10-12-evaluation')
sys.path.insert(0,str(ROOT));import nmr_helpers
# No raw/test structures or old test predictions are permitted in this evaluation.
def audit(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  name=Path(os.fsdecode(args[0])).name
  if name in ('test.xyz','raw.xyz'):raise RuntimeError('Evaluation forbids raw/test data access')
sys.addaudithook(audit)
torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
RUNS={'09':'TensorNequIP-NMR-FineTune-9086968','10':'TensorNequIP-NMR-Scratch-LR1e-4-9115507','11':'TensorNequIP-NMR-FrozenTransfer-9115508','12':'TensorNequIP-NMR-FrozenTransfer-LR3e-3-9115945'}
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def same(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
source=torch.load(ROOT/'graph-pes-results/Kavanagh-Med-9059603/model.pt',map_location='cpu',weights_only=False)
models={};provenance=[];configs={}
for run,folder in RUNS.items():
 d=ROOT/'graph-pes-results'/folder
 cfg=yaml.safe_load((d/'train-config.yaml').read_text());configs[run]=cfg
 assert cfg['data']['test'] is None
 assert cfg['data']['train']['path']=='data/nmr_iso/train.xyz' and cfg['data']['valid']['path']=='data/nmr_iso/valid.xyz'
 model=torch.load(d/'model.pt',map_location='cpu',weights_only=False)
 ck=torch.load(d/'checkpoints/best.ckpt',map_location='cpu',weights_only=False)
 last=torch.load(d/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
 state={k.removeprefix('model.'):v for k,v in ck['state_dict'].items() if k.startswith('model.')}
 matches=same(model.state_dict(),state)
 if not matches:
  assert run=='11','Unexpected artifact mismatch'
  models['11-export']=copy.deepcopy(model).eval().requires_grad_(False)
 model.load_state_dict(state,strict=True);models[run]=model.eval().requires_grad_(False)
 summary=yaml.safe_load((d/'summary.yaml').read_text())
 selected=[v for k,v in last['callbacks'].items() if k.startswith('ModelCheckpoint') and v.get('monitor')=='valid/loss/total'][0]
 frozen_equal=None
 if run in ('11','12'):
  frozen_equal=all(same(getattr(model.models['nequip'],part).state_dict(),getattr(source.models['nequip'],part).state_dict()) for part in ('Z_embedding','initial_node_embedding','edge_embedding','layers'))
  assert frozen_equal
 record={'run':run,'directory':folder,'best_epoch':int(ck['epoch'])+1,'last_epoch':int(last['epoch'])+1,'export_matches_checkpoint':matches,'checkpoint_selection_loss':float(selected['best_model_score']),'summary_best_loss':float(summary['early_stopping/best_valid_loss']),'runtime_minutes':float(summary['_runtime'])/60,'initial_lr':cfg['fitting']['optimizer']['lr'],'final_lr':summary['lr-AdamW/normal'],'trainable_parameters':summary['n_learnable_parameters'],'frozen_backbone_matches_EF':frozen_equal,'model_sha256':sha(d/'model.pt'),'checkpoint_sha256':sha(d/'checkpoints/best.ckpt'),'last_checkpoint_sha256':sha(d/'checkpoints/last.ckpt'),'config_sha256':sha(d/'train-config.yaml')}
 provenance.append(record);print(json.dumps(record),flush=True)
 del ck,last
assert configs['09']['data']==configs['10']['data']
for key in ['loss','fitting']:
 assert configs['09'][key]==configs['10'][key],key
assert configs['11']['model']==configs['12']['model']
# The only scientific fitting difference between 11 and 12 is initial LR.
a=copy.deepcopy(configs['11']['fitting']);b=copy.deepcopy(configs['12']['fitting']);a['optimizer']['lr']=b['optimizer']['lr'];assert a==b
provenance=pd.DataFrame(provenance);provenance.to_csv(OUT/'provenance.csv',index=False)
frames={s:read(ROOT/f'data/nmr_iso/{s}.xyz',index=':') for s in ['train','valid']}
assert len(frames['train'])==800 and len(frames['valid'])==50
records=[];data_hashes={};reference_rows=[]
for split,atoms in frames.items():
 data_hashes[split]=sha(ROOT/f'data/nmr_iso/{split}.xyz')
 for i,a in enumerate(atoms):
  y=np.asarray(a.arrays['MS_Iso_QM'],dtype=float)
  np.testing.assert_allclose(y,np.trace(a.arrays['ms'].reshape(-1,3,3),axis1=1,axis2=2)/3,rtol=0,atol=1e-6)
  reference_rows.append(pd.DataFrame({'split':split,'structure':i,'atom':np.arange(len(a)),'element':a.get_chemical_symbols(),'reference':y}))
geometry=pd.concat(reference_rows,ignore_index=True)
# Shared frozen features can be reused only after exact backbone equality above.
feature={}
hook=models['11'].models['nequip'].tensor_readout.register_forward_pre_hook(lambda module,args:feature.update(h=args[0]))
for split in ['valid','train']:
 pred={r:[] for r in ['09','10','11','12','11-export']};started=time.perf_counter()
 for i,a in enumerate(frames[split]):
  graph=AtomicGraph.from_ase(a,cutoff=6.)
  assert not graph.properties and not graph.other
  with torch.inference_mode():
   for r in ['09','10','11']:
    pred[r].append(models[r].predict(graph,['tensor'])['tensor'].cpu().numpy().copy())
   for r in ['12','11-export']:
    m=models[r];net=m.models['nequip']
    val=net.scaler(net.tensor_readout(feature['h']),graph)+m.models['offset'](graph)['tensor']
    pred[r].append(val.cpu().numpy().copy())
    if split=='valid' and i in [0,25,49]:
     direct=m.predict(graph,['tensor'])['tensor'].cpu().numpy()
     np.testing.assert_allclose(pred[r][-1],direct,rtol=1e-6,atol=1e-5)
  if (i+1)%25==0:print(f'{split}: {i+1}/{len(frames[split])} structures, {time.perf_counter()-started:.1f}s',flush=True)
 for r,ys in pred.items():
  df=geometry[geometry.split==split].copy();df['run']=r;df['prediction']=np.concatenate(ys)
  assert np.isfinite(df.prediction).all()
  df['error']=df.prediction-df.reference;records.append(df)
 current=pd.concat(records,ignore_index=True);current.to_csv(OUT/'sites.csv.gz',index=False)
 if split=='valid':
  for row in provenance.to_dict('records'):
   v=current[current.run==row['run']];loss=float(np.mean(v.error**2)*1000)
   assert np.isclose(loss,row['checkpoint_selection_loss'],rtol=.005),(row['run'],loss,row['checkpoint_selection_loss'])
  sanity=current[(current.run=='11-export')&(current.structure<2)]
  print('11 exported model first-two-structure loss:',float(np.mean(sanity.error**2)*1000),flush=True)
  print('VALIDATION METRICS',current.groupby(['run','element']).error.agg(MAE=lambda x:np.mean(np.abs(x)),RMSE=lambda x:np.sqrt(np.mean(x*x))).to_string(),flush=True)
hook.remove()
manifest={'data_sha256':data_hashes,'EF_source_sha256':sha(ROOT/'graph-pes-results/Kavanagh-Med-9059603/model.pt'),'counts':{'train_structures':800,'valid_structures':50,'train_sites':115200,'valid_sites':7200},'test_accessed':False,'torch':torch.__version__,'numpy':np.__version__,'inference':'CPU float32; metrics float64; four threads','frozen_inference':'Shared features after exact backbone equality; direct predictions cross-checked on validation structures 0,25,49.','primary_artifact':'checkpoints/best.ckpt loaded strictly into model.pt architecture; exported 11 additionally evaluated','evaluator_sha256':sha(__file__)}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('ALL INFERENCE COMPLETE',flush=True)
