"""Audit validation-selected checkpoints and evaluate the matched NMR learning curves.

Run with the research repository's Python environment. No fitting or test selection.
Existing predictions are reused only after validating their data/checkpoint hashes.
"""
from pathlib import Path
import os, sys, json, hashlib, time, gc
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nmr-comparison-mpl')
sys.dont_write_bytecode=True
ROOT=Path(os.environ.get('NEQUIP_PROJECT','/Users/louismcauliffe/code/research/NequIP_experiments'))
DEFAULT_OUT=Path(__file__).resolve().parent
if not DEFAULT_OUT.name.endswith('-data'): DEFAULT_OUT=DEFAULT_OUT/'data'
OUT=Path(os.environ.get('NMR_COMPARISON_OUTPUT',str(DEFAULT_OUT)))
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import yaml, torch
from ase.io import read
from graph_pes.atomic_graph import AtomicGraph
from graph_pes.utils.nn import count_used_parameters
import nmr_helpers, nmr_helpers.elora, nmr_helpers.frozen_head_training
torch.set_num_threads(4)
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high')
METHODS={17:'Direct training',19:'Last layer + NMR head',21:'NMR head only',22:'ELoRA',23:'Full fine-tuning'}
GROUPS={17:'17-Scratch-LearningCurve',19:'19-LastLayerTransfer-LearningCurve',21:'21-FrozenBackbone-LearningCurve',22:'22-ELoRA-LearningCurve',23:'23-FullFineTune-LearningCurve'}
SIZES=[10,50,100,200,800]
STDS={'Si':8.789599019755547,'O':19.197689920683118}
PILOT='22-ELoRA-N100-R4-9201369'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes={}
def track(p):
    hashes[str(p)]=sha(p)
    return hashes[str(p)]
def equal(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def score(y,p,z):
    result={}
    for num,e in [(14,'Si'),(8,'O')]:
        ref=y[z==num]; err=p[z==num].astype(float)-ref
        result[e+'_rmse']=float(np.sqrt(np.mean(err**2)))
        result[e+'_mae']=float(np.mean(np.abs(err)))
        result[e+'_r2']=float(1-np.sum(err**2)/np.sum((ref-ref.mean())**2))
        result[e+'_standardized_mse']=(result[e+'_rmse']/STDS[e])**2
    result['loss']=.5*sum(result[e+'_standardized_mse'] for e in STDS)
    return result

OLD=ROOT/'experiments/21-Test-DataEfficiency-data'
old_manifest=json.loads((OLD/'manifest.json').read_text())
old_rows=pd.read_csv(OLD/'metrics.csv')
cached={s:dict(np.load(OLD/f'{s}-predictions.npz')) for s in ('valid','test')}
frames={}
arrays={}
for s in ('train','valid','test'):
    p=ROOT/f'data/nmr_iso/{s}.xyz'
    assert track(p)==old_manifest['input_sha256'][str(p)]
    frames[s]=read(p,':')
assert [len(frames[s]) for s in ('train','valid','test')]==[800,50,150]
for s in ('valid','test'):
    aa=frames[s]
    arrays[s]={'reference':np.concatenate([a.arrays['MS_Iso_QM'].reshape(-1) for a in aa]).astype(float),
               'atomic_numbers':np.concatenate([a.numbers for a in aa]),
               'structure':np.concatenate([np.full(len(a),i) for i,a in enumerate(aa)])}
    for key in arrays[s]:
        np.testing.assert_array_equal(arrays[s][key],cached[s][key])
graphs={s:[AtomicGraph.from_ase(a,cutoff=6.) for a in frames[s]] for s in ('valid','test')}
assert all(not g.properties and not g.other for gg in graphs.values() for g in gg)
rows=[]; metrics=[]; structure_metrics=[]
started=time.monotonic()
resume={}
resume_file=OUT/'resume-manifest.json'
if resume_file.exists() and os.environ.get('NMR_RESUME_PARTIAL')=='1':
    resume_manifest=json.loads(resume_file.read_text())
    assert all(sha(p)==h for p,h in resume_manifest['input_sha256'].items())
    resume_dir=OUT/resume_manifest['cache_directory']
    assert all(sha(resume_dir/name)==h for name,h in resume_manifest['cache_sha256'].items())
    resume={s:dict(np.load(resume_dir/f'{s}-predictions.npz')) for s in ('valid','test')}
    for s in resume:
        for key in ('reference','atomic_numbers','structure'):
            np.testing.assert_array_equal(resume[s][key],arrays[s][key])
for exp,group in GROUPS.items():
    dirs=sorted((ROOT/'graph-pes-results'/group).glob('*/train-config.yaml'))
    assert len(dirs)==(6 if exp==22 else 5)
    for cp in dirs:
        d=cp.parent; rid=d.name; cfg=yaml.safe_load(cp.read_text()); n=cfg['data']['train']['n']
        assert cfg['data']['test'] is None
        assert cfg['general']['seed']==cfg['data']['train']['seed']==42
        assert cfg['data']['train']['path']=='data/nmr_iso/train.xyz' and cfg['data']['train']['shuffle']
        assert cfg['data']['valid']['path']=='data/nmr_iso/valid.xyz'
        lc=cfg['loss']['isotropy']['+nmr_helpers.shared_head_training.ElementBalancedMSE']
        assert lc['si_std']==STDS['Si'] and lc['o_std']==STDS['O'] and lc['weight']==1
        inputs=[cp,d/'model.pt',d/'checkpoints/best.ckpt',d/'checkpoints/last.ckpt']
        if (d/'summary.yaml').exists(): inputs.append(d/'summary.yaml')
        for p in inputs:
            digest=track(p)
            if exp in (17,19,21):
                original=p if exp!=21 else p.parent
                original=ROOT/'graph-pes-results'/rid/p.relative_to(d) if exp==21 else p
                assert digest==old_manifest['input_sha256'][str(original)], (rid,p)
        best=torch.load(d/'checkpoints/best.ckpt',map_location='cpu',weights_only=False)
        last=torch.load(d/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        model=torch.load(d/'model.pt',map_location='cpu',weights_only=False)
        weights={k.removeprefix('model.'):v for k,v in best['state_dict'].items() if k.startswith('model.')}
        matches=equal(model.state_dict(),weights)
        assert matches,rid
        model.load_state_dict(weights,strict=True)
        early=next(v for k,v in last['callbacks'].items() if k.startswith('EarlyStopping'))
        selected=next(v for k,v in last['callbacks'].items() if k.startswith('ModelCheckpoint') and v.get('monitor')=='valid/loss/total')
        best_score=float(selected['best_model_score'])
        checks=int(early['total_checks']-early['best_val_loss_check'])
        best_epoch=int(best['epoch'])+1; final_epoch=int(last['epoch'])+1
        assert best_epoch==int(early['best_val_loss_check'])
        status='Early stopping' if checks>=cfg['fitting']['early_stopping']['patience'] else 'Epoch ceiling' if final_epoch>=cfg['fitting']['trainer_kwargs']['max_epochs'] else 'Stopped before early-stopping threshold'
        # Element-access metadata is empty in some serialised exports; restore
        # the known Si/O training species before counting active parameters.
        for parameter in model.parameters():
            if hasattr(parameter,'register_elements'): parameter.register_elements([8,14])
        summary=yaml.safe_load((d/'summary.yaml').read_text()) if (d/'summary.yaml').exists() else {}
        stopping_status=status
        convergence_assessment=''
        if rid=='22-ELoRA-N800-R4-9201983':
            status='Converged (researcher assessment)'
            convergence_assessment='Researcher confirms practical convergence: the learning rate had decayed too far for further meaningful improvement to be expected. Recorded on 2026-10-29; distinct from formal early-stopping termination.'
        row=dict(run_id=rid,experiment=exp,method=METHODS[exp],n=n,primary=rid!=PILOT,
                 best_epoch=best_epoch,final_epoch=final_epoch,status=status,stopping_status=stopping_status,convergence_assessment=convergence_assessment,checks_since_best=checks,
                 saved_valid_loss=best_score,export_matches_checkpoint=matches,
                 trainable_parameters=summary.get('n_learnable_parameters',count_used_parameters(model,only_learnable=True)),optimizer_config=json.dumps(cfg['fitting']['optimizer'],sort_keys=True),
                 runtime_minutes=summary.get('_runtime',float('nan'))/60,result_directory=str(d),
                 predictions_reused=exp in (17,19,21) or rid in resume.get('test',{}))
        rows.append(row)
        print('AUDIT',rid,status,'best',best_epoch,'last',final_epoch,flush=True)
        model.eval().requires_grad_(False)
        for split in ('valid','test'):
            if exp in (17,19,21):
                prediction=cached[split][rid]
            elif rid in resume.get(split,{}):
                prediction=resume[split][rid]
                print('REUSE AUDITED PARTIAL',rid,split,flush=True)
            else:
                preds=[]
                with torch.inference_mode():
                    for i,g in enumerate(graphs[split]):
                        preds.append(model.predict(g,['tensor'])['tensor'].cpu().numpy().reshape(-1).copy())
                        if (i+1)%50==0:
                            print('PREDICT',rid,split,i+1,'elapsed',round(time.monotonic()-started),flush=True)
                prediction=np.concatenate(preds).astype(float)
            a=arrays[split]; y=a['reference']; z=a['atomic_numbers']
            assert prediction.shape==y.shape and np.isfinite(prediction).all()
            a[rid]=prediction
            scored=score(y,prediction,z)
            if split=='valid':
                # CPU and GPU float32 reductions differ slightly; weights match exactly.
                assert np.isclose(scored['loss'],best_score,rtol=1e-3,atol=1e-7),(rid,scored['loss'],best_score)
            metrics.append({**row,'split':split,**scored})
            for j in range(len(frames[split])):
                mask=a['structure']==j
                structure_metrics.append(dict(run_id=rid,experiment=exp,n=n,split=split,structure=j,
                    **score(y[mask],prediction[mask],z[mask])))
        pd.DataFrame(rows).to_csv(OUT/'checkpoint-audit.csv',index=False)
        pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
        for split in arrays: np.savez_compressed(OUT/f'{split}-predictions.npz',**arrays[split])
        del model,best,last,weights
        gc.collect()

df=pd.DataFrame(metrics)
for exp in METHODS:
    assert df.query('experiment==@exp and primary and split=="test"').n.sort_values().tolist()==SIZES
pd.DataFrame(structure_metrics).to_csv(OUT/'structure-metrics.csv',index=False)
for s in ('valid','test'):
    points=df.query('primary and split==@s').copy()
    points['log10_n']=np.log10(points.n)
    for m in ('loss','Si_rmse','O_rmse'): points['log10_'+m]=np.log10(points[m])
    points.to_csv(OUT/f'{s}-plotted-data.csv',index=False)
assert all(sha(p)==h for p,h in hashes.items()),'Input changed during evaluation'
manifest=dict(date='2026-10-29',input_sha256=hashes,evaluator_sha256=sha(__file__),
    methods=METHODS,sizes=SIZES,counts={s:len(frames[s]) for s in frames},loss_stds=STDS,
    checkpoint_selection='Minimum validation loss; no test-based checkpoint, run, or hyperparameter selection',
    primary_elora_n100='22-ELoRA-N100-R4-9201981',excluded_from_primary_curve=PILOT,
    duplicate_policy='Use the N100 run submitted with the five-size sweep; retain earlier pilot separately. Chosen before test evaluation.',
    reused_cache=str(OLD),cache_manifest_sha256=sha(OLD/'manifest.json'),
    cache_prediction_sha256={s:sha(OLD/f'{s}-predictions.npz') for s in ('valid','test')},
    reuse_checks='Exact current model, best/last checkpoint, config, summary and dataset hashes match original manifest; prediction references and atom ordering checked.',
    geometry_overlap=old_manifest['geometry_overlap'],ef_pretraining_test_geometry_overlap=old_manifest['ef_pretraining_test_geometry_overlap'],
    inference='CPU float32, four threads; metrics float64; labels not passed into models',
    validation_reproduction_tolerance='0.1% relative plus 1e-7 absolute, allowing CPU/GPU float32 numerical drift; checkpoint weights must match exactly',
    seed=42,repetitions=1,uncertainty='No seed error bars: one sweep per method. ELoRA pilot is not an independent seed.',
    fairness_note='Same nested training subsets and held-out splits; method-specific learning rates and stopping budgets. N excludes 50 validation structures. Fixed loss denominators use full training pool.',
    partial_evaluation_resume=bool(resume),elapsed_seconds=time.monotonic()-started)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(df.query('primary and split=="test"')[['experiment','n','loss','Si_rmse','O_rmse','status']].to_string(index=False),flush=True)
print('EVALUATION COMPLETE; INPUT HASHES UNCHANGED',flush=True)
