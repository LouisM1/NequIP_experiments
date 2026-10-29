"""Evaluate validation-selected checkpoints; never train or select using test data."""
from pathlib import Path
import gc, hashlib, json, os, sys, time

ROOT = Path('/Users/louismcauliffe/code/research/NequIP_experiments')
OUT = Path(os.environ.get('NMR_TEST_EVAL_OUTPUT', str(Path(__file__).resolve().parent)))
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/nmr-test-eval-mpl')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from ase.io import read
from graph_pes.atomic_graph import AtomicGraph
import nmr_helpers
from nmr_helpers.frozen_head_training import CachedTensorNequIP, BACKBONE_PARTS

torch.set_num_threads(4)
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
STDS = {'Si': 8.789599019755547, 'O': 19.197689920683118}
METHODS = {17: 'Scratch', 18: 'Frozen: original', 19: 'Final layer + head', 21: 'Frozen: improved'}
hashes = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def track(p):
    hashes[str(p)] = sha(p)

def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b

def geometry_key(a):
    # Explicitly an ordered-geometry check; not a near-duplicate/trajectory audit.
    h = hashlib.sha256()
    for arr in (a.numbers, np.round(a.positions, 6), np.round(a.cell.array, 6), a.pbc):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()

def score(y, pred, z):
    result = {}
    for number, elem in ((14, 'Si'), (8, 'O')):
        ref, pr = y[z == number], pred[z == number]
        err = pr.astype(np.float64) - ref.astype(np.float64)
        mse = float(np.mean(err**2))
        result[elem+'_rmse'] = float(np.sqrt(mse))
        result[elem+'_mae'] = float(np.mean(np.abs(err)))
        result[elem+'_r2'] = float(1-np.sum(err**2)/np.sum((ref-ref.mean())**2))
    result['loss'] = .5*sum((result[e+'_rmse']/STDS[e])**2 for e in STDS)
    return result

frames = {}
for split in ('train', 'valid', 'test'):
    p = ROOT/f'data/nmr_iso/{split}.xyz'
    track(p)
    frames[split] = read(p, ':')
    for a in frames[split]:
        y = np.asarray(a.arrays['MS_Iso_QM'], dtype=np.float64).reshape(-1)
        assert len(y) == len(a) and np.isfinite(y).all()
        np.testing.assert_allclose(y, np.trace(a.arrays['ms'].reshape(-1,3,3),axis1=1,axis2=2)/3, rtol=0, atol=1e-6)
assert [len(frames[s]) for s in ('train','valid','test')] == [800,50,150]
geom = {s:set(map(geometry_key,aa)) for s,aa in frames.items()}
overlaps = {a+'_'+b:len(geom[a]&geom[b]) for a,b in [('train','test'),('valid','test'),('train','valid')]}
assert not any(overlaps.values()), overlaps
ef_overlap = {}
for split in ('train','valid'):
    p = ROOT/f'data/{split}.xyz'
    if p.exists():
        track(p)
        ef_overlap[split] = len(set(map(geometry_key,read(p, ':')))&geom['test'])
source_path = ROOT/'graph-pes-results/Kavanagh-Med-9059603/model.pt'
track(source_path)
source = torch.load(source_path, map_location='cpu', weights_only=False).models['nequip']
models, rows = {}, []
paths = sorted(ROOT.glob('graph-pes-results/1[789]*/*/train-config.yaml'))
paths += sorted(ROOT.glob('graph-pes-results/21-FrozenBackbone-LearningCurve/*/train-config.yaml'))
assert len(paths) == 20, len(paths)
for cp in paths:
    d = cp.parent
    cfg = yaml.safe_load(cp.read_text())
    run_id = d.name
    exp = int(run_id[:2]); n = cfg['data']['train']['n']
    assert cfg['general']['seed'] == cfg['data']['train']['seed'] == 42
    assert cfg['data']['train']['path'] == 'data/nmr_iso/train.xyz'
    assert cfg['data']['train']['shuffle']
    assert cfg['data']['valid']['path'] == 'data/nmr_iso/valid.xyz'
    assert cfg['data']['test'] is None
    lc = cfg['loss']['isotropy']['+nmr_helpers.shared_head_training.ElementBalancedMSE']
    assert lc['si_std'] == STDS['Si'] and lc['o_std'] == STDS['O'] and lc['weight'] == 1
    for p in (cp, d/'model.pt', d/'checkpoints/best.ckpt', d/'checkpoints/last.ckpt'):
        track(p)
    best = torch.load(d/'checkpoints/best.ckpt',map_location='cpu',weights_only=False)
    last = torch.load(d/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    model = torch.load(d/'model.pt',map_location='cpu',weights_only=False)
    weights = {k.removeprefix('model.'):v for k,v in best['state_dict'].items() if k.startswith('model.')}
    export_matches = same(model.state_dict(),weights)
    model.load_state_dict(weights, strict=True)
    early = next(v for k,v in last['callbacks'].items() if k.startswith('EarlyStopping'))
    selected = next(v for k,v in last['callbacks'].items() if k.startswith('ModelCheckpoint') and v.get('monitor')=='valid/loss/total')
    best_score = float(selected['best_model_score'])
    assert np.isclose(best_score,float(early['best_val_loss']),rtol=1e-7)
    best_epoch = int(best['epoch'])+1; final_epoch = int(last['epoch'])+1
    assert best_epoch == int(early['best_val_loss_check'])
    checks = int(early['total_checks']-early['best_val_loss_check'])
    ceiling = cfg['fitting']['trainer_kwargs']['max_epochs']
    status = ('Early stopping' if checks>=cfg['fitting']['early_stopping']['patience'] else
              'Epoch ceiling' if final_epoch>=ceiling else 'Stopped before convergence')
    nq = model.models['nequip']
    assert nq.tensor_readout.weight.numel()==128
    if exp in (18,19,21):
        for part in BACKBONE_PARTS[:-1]:
            assert same(getattr(nq,part).state_dict(),getattr(source,part).state_dict()), (run_id,part)
        for layer in range(3 if exp==19 else 4):
            assert same(nq.layers[layer].state_dict(),source.layers[layer].state_dict()), (run_id,layer)
        if exp==19:
            assert not same(nq.layers[3].state_dict(),source.layers[3].state_dict())
    summary = yaml.safe_load((d/'summary.yaml').read_text()) if (d/'summary.yaml').exists() else {}
    if summary:
        track(d/'summary.yaml')
    model.eval().requires_grad_(False)
    models[run_id] = model
    row = dict(run_id=run_id,experiment=exp,method=METHODS[exp],n=n,best_epoch=best_epoch,
               final_epoch=final_epoch,optimizer_steps=int(last['global_step']),checks_since_best=checks,
               status=status,provisional=status!='Early stopping',saved_valid_loss=best_score,
               export_matches_checkpoint=export_matches,frozen_layers_verified=exp in (18,19,21),
               active_trainable_parameters={17:738052,18:132,19:141380,21:132}[exp],
               runtime_minutes=summary.get('_runtime',float('nan'))/60,
               result_directory=str(d))
    rows.append(row)
    print('AUDIT',run_id,best_epoch,final_epoch,status,'export_matches',export_matches,flush=True)
    del best,last,weights
    gc.collect()
assert all(sum(r['experiment']==e and r['n']==n for r in rows)==1 for e in METHODS for n in (10,50,100,200,800))
pd.DataFrame(rows).to_csv(OUT/'checkpoint-audit.csv',index=False)
frozen_ids = [r['run_id'] for r in rows if r['experiment'] in (18,21)]
full_ids = [r['run_id'] for r in rows if r['experiment'] in (17,19)]
anchor_id = next(r['run_id'] for r in rows if r['experiment']==21)
anchor = models[anchor_id].models['nequip']
all_metrics = []; structures = []
started = time.monotonic()
for split in ('valid','test'):
    aa = frames[split]
    target = np.concatenate([a.arrays['MS_Iso_QM'].reshape(-1) for a in aa]).astype(np.float64)
    z = np.concatenate([a.numbers for a in aa])
    structure_id = np.concatenate([np.full(len(a),i) for i,a in enumerate(aa)])
    predictions = {k:[] for k in models}
    for i,a in enumerate(aa):
        g = AtomicGraph.from_ase(a,cutoff=6.)
        assert not g.properties and not g.other
        with torch.inference_mode():
            h = anchor.frozen_features(g)
            for rid in frozen_ids:
                m = models[rid]; net = m.models['nequip']
                y = net.scaler(net.tensor_readout(h),g)+m.models['offset'](g)['tensor']
                predictions[rid].append(y.cpu().numpy().reshape(-1).copy())
                if i in (0,len(aa)//2,len(aa)-1):
                    direct = m.predict(g,['tensor'])['tensor'].cpu().numpy().reshape(-1)
                    np.testing.assert_allclose(direct,predictions[rid][-1],rtol=1e-6,atol=1e-5)
            for rid in full_ids:
                predictions[rid].append(models[rid].predict(g,['tensor'])['tensor'].cpu().numpy().reshape(-1).copy())
        if (i+1)%10==0:
            print('INFERENCE',split,i+1,'/',len(aa),'elapsed',round(time.monotonic()-started,1),flush=True)
    arrays = {'reference':target,'atomic_numbers':z,'structure':structure_id}
    for row in rows:
        rid = row['run_id']; pr = np.concatenate(predictions[rid]).astype(np.float64)
        assert pr.shape==target.shape and np.isfinite(pr).all()
        arrays[rid] = pr
        metrics = score(target,pr,z)
        if split=='valid':
            assert np.isclose(metrics['loss'],row['saved_valid_loss'],rtol=5e-4,atol=1e-7), (rid,metrics['loss'],row['saved_valid_loss'])
        all_metrics.append({**row,'split':split,**metrics})
        for j in range(len(aa)):
            mask=structure_id==j
            structures.append(dict(run_id=rid,experiment=row['experiment'],n=row['n'],split=split,structure=j,
                                   **score(target[mask],pr[mask],z[mask])))
    np.savez_compressed(OUT/f'{split}-predictions.npz',**arrays)
    pd.DataFrame(all_metrics).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(structures).to_csv(OUT/'structure-metrics.csv',index=False)
    print('SPLIT COMPLETE',split,flush=True)
test = pd.DataFrame(all_metrics).query("split=='test'")
paired=[]
rng=np.random.default_rng(20261028)
draws=rng.integers(0,len(frames['test']),size=(2000,len(frames['test'])))
st=pd.DataFrame(structures).query("split=='test'")
for n in (10,50,100,200,800):
    a=st.query('experiment==21 and n==@n').sort_values('structure')
    for e in (17,18,19):
        b=st.query('experiment==@e and n==@n').sort_values('structure')
        record=dict(n=n,comparator=e)
        for elem in ('Si','O'):
            # All structures have the same elemental counts; pooled RMSE equals sqrt(mean(structure MSE)).
            x=a[elem+'_rmse'].to_numpy()**2; y=b[elem+'_rmse'].to_numpy()**2
            delta=np.sqrt(x[draws].mean(axis=1))-np.sqrt(y[draws].mean(axis=1))
            record[elem+'_rmse_delta']=float(np.sqrt(x.mean())-np.sqrt(y.mean()))
            record[elem+'_delta_low'],record[elem+'_delta_high']=map(float,np.quantile(delta,[.025,.975]))
        paired.append(record)
pd.DataFrame(paired).to_csv(OUT/'paired-test-comparisons.csv',index=False)
manifest=dict(date='2026-10-28',checkpoint_selection='Minimum validation loss, never test loss',
              input_sha256=hashes,evaluator_sha256=sha(__file__),
              counts={s:dict(structures=len(aa),atoms=sum(len(a) for a in aa),
                           Si=sum(int((a.numbers==14).sum()) for a in aa),O=sum(int((a.numbers==8).sum()) for a in aa)) for s,aa in frames.items()},
              geometry_overlap=overlaps,ef_pretraining_test_geometry_overlap=ef_overlap,
              overlap_check='Ordered atomic numbers, positions and cell rounded to 1e-6 angstrom, PBC. Does not exclude correlated/near-duplicate structures.',
              inference='CPU float32, four threads; metrics float64. Fresh predictions, no labels passed to models.',
              frozen_optimisation='Shared freshly computed features after exact backbone equality; full forwards cross-checked on first/middle/last structure of each split for every frozen model.',
              test_used_for_training_or_selection=False,seed=42,repetitions=1,loss_stds=STDS,
              data_budget_note='N fitting structures plus 50 validation structures; fixed loss denominators inherited from 800-structure training pool.',
              bootstrap='2000 paired whole-structure resamples; sample uncertainty only, not seed variation. Equal elemental counts per structure.',
              elapsed_inference_seconds=time.monotonic()-started)
for aa in frames.values():
    assert all((a.numbers==14).sum()==48 and (a.numbers==8).sum()==96 for a in aa)
assert all(sha(p)==h for p,h in hashes.items()), 'An input changed during evaluation'
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(test[['experiment','n','loss','Si_rmse','O_rmse','Si_mae','O_mae','status']].sort_values(['experiment','n']).to_string(index=False),flush=True)
print('ALL CHECKS PASSED; INPUTS UNCHANGED',flush=True)
