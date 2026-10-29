"""Validate all plotted values and use the training logger's active-parameter count."""
from pathlib import Path
import os,sys,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nmr-comparison-mpl')
sys.dont_write_bytecode=True
ROOT=Path('/Users/louismcauliffe/code/research/NequIP_experiments')
HERE=Path(__file__).resolve().parent
DATA=HERE if HERE.name.endswith('-data') else HERE/'data'
sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
import nmr_helpers,nmr_helpers.elora
from graph_pes.utils.nn import count_used_parameters
rid='22-ELoRA-N800-R4-9201983'
model=torch.load(ROOT/'graph-pes-results/22-ELoRA-LearningCurve'/rid/'model.pt',map_location='cpu',weights_only=False)
for p in model.parameters():
    if hasattr(p,'register_elements'): p.register_elements([8,14])
count=count_used_parameters(model,only_learnable=True)
assert count==29060
for name in ['checkpoint-audit.csv','metrics.csv','test-plotted-data.csv','valid-plotted-data.csv']:
    p=DATA/name; frame=pd.read_csv(p)
    frame.loc[frame.run_id==rid,'trainable_parameters']=count
    frame.to_csv(p,index=False)
metrics=pd.read_csv(DATA/'metrics.csv')
assert len(metrics)==52 and len(metrics.query('primary'))==50
assert metrics.query('primary and status != "Early stopping"').run_id.unique().tolist()==[rid]
for split in ('valid','test'):
    pred=dict(np.load(DATA/f'{split}-predictions.npz'))
    y=pred['reference'];z=pred['atomic_numbers']
    for row in metrics.query('split==@split').itertuples():
        loss=0
        for number,element,std in [(14,'Si',8.789599019755547),(8,'O',19.197689920683118)]:
            err=pred[row.run_id][z==number]-y[z==number]
            rmse=np.sqrt(np.mean(err**2))
            np.testing.assert_allclose(rmse,getattr(row,element+'_rmse'),rtol=1e-12)
            loss+=.5*(rmse/std)**2
        np.testing.assert_allclose(loss,row.loss,rtol=1e-12)
manifest=json.loads((DATA/'manifest.json').read_text())
manifest['parameter_count_note']='Active trainable parameters use graph_pes.utils.nn.count_used_parameters, as in the training logger; inactive elemental entries excluded. N800 ELoRA has 29,060 active trainable parameters.'
manifest['final_validation']='All 52 evaluations independently recalculated from saved site predictions; 25 primary test points, 25 primary validation points, plus pilot on each split.'
manifest['finalizer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
manifest['reproduction_evaluator_sha256']=hashlib.sha256((HERE/'evaluate.py').read_bytes()).hexdigest()
(DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Verified all 52 evaluations against saved predictions. Active ELoRA parameter count:',count)
print(metrics.query('primary and split=="test"').pivot(index='n',columns='method',values='loss').to_string())
