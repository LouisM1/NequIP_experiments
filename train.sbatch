#!/bin/bash
# Run these from the project directory containing data/ and configs/:
#   mkdir -p logs
#   sbatch train.sbatch configs/06-Kavanagh-Med-HyperParams.yaml Kavanagh-Med
# Resume THIS new run after a time limit (use its original results directory):
#   sbatch train.sbatch --resume graph-pes-results/Kavanagh-Med-ORIGINAL_JOB_ID

#SBATCH --job-name=nequip-medium
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/experiment-%j.out
#SBATCH --error=logs/experiment-%j.err

# No explicit memory request: ARC documents a default of 8000 MB per CPU.
# Eight CPUs therefore normally give about 64 GB of SYSTEM RAM, not GPU RAM.
# The actual allocation is printed below. Memory does not grow automatically.
# Medium/L40S access remains subject to the current ARC partition/account rules.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch}"

if [[ $# -ne 2 ]]; then
    echo 'Usage: sbatch train.sbatch CONFIG.yaml RUN_NAME'
    echo '   or: sbatch train.sbatch --resume PREVIOUS_RUN_DIRECTORY'
    exit 2
fi

if [[ "$1" == --resume ]]; then
    TRAIN_DIR="$2"
    for required in train-config.yaml checkpoints/last.ckpt; do
        [[ -f "$TRAIN_DIR/$required" ]] || {
            echo "Missing resume file: $TRAIN_DIR/$required"
            exit 1
        }
    done
else
    CONFIG="$1"
    RUN_NAME="$2"
    [[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG"; exit 1; }
    [[ "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || {
        echo 'Use only letters, numbers, underscores and hyphens in RUN_NAME.'
        exit 2
    }
fi

module purge
module load Anaconda3/2025.06-1
conda activate /data/oums-atom-elec-ml/trin4011/envs/graph-pes-arc
PYTHON=/data/oums-atom-elec-ml/trin4011/envs/graph-pes-arc/bin/python

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo 'Allocated resources:'
scontrol show job "$SLURM_JOB_ID"

# Keep the compatibility fixes local to this process; no installed files change.
srun "$PYTHON" - "$1" "$2" <<'TRAIN_PY'
import os
import sys
from pathlib import Path
from importlib.metadata import version
import torch
import yaml
from graph_pes.config.shared import instantiate_config_from_dict, parse_model, parse_loss
from graph_pes.config.training import TrainingConfig
from graph_pes.data.loader import GraphDataLoader
from graph_pes.scripts.train import trainer_from_config
from nmr_helpers.shared_head_training import train_from_config
from graph_pes.scripts.utils import configure_general_options, get_data_from_cli_arg
from graph_pes.training.opt import Optimizer
from graph_pes.training.tasks import TrainingTask, get_all_eval_metrics
from graph_pes.utils.logger import log_to_file
from graph_pes.utils.misc import nested_merge_all

print("Python:", sys.executable)
print("graph-pes:", version("graph-pes"))
print("Lightning:", version("pytorch-lightning"))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "No usable GPU in this allocation"
gpu = torch.cuda.get_device_properties(0)
print("GPU:", gpu.name, "VRAM (GiB):", round(gpu.total_memory / 2**30, 1))
print("GPU calculation:", (torch.ones(4, device="cuda") * 2).tolist())

# graph-pes 1.0.2 builds optimizer groups using sets of Parameters.
# Their order can change between processes, misassigning saved Adam state.
# Use stable parameter-name order in BOTH fresh training and resumption,
# preserving the same two weight-decay groups and optimizer settings.
def ordered_optimizer(self, model):
    named = sorted(model.named_parameters())
    non_decayable = {id(p) for p in model.non_decayable_parameters()}
    groups = []
    for label, use_non_decay in [("non-decayable", True), ("normal", False)]:
        members = [(n, p) for n, p in named if (id(p) in non_decayable) == use_non_decay]
        if members:
            group = dict(name=label, params=[p for _, p in members],
                         param_names=[n for n, _ in members])
            if use_non_decay:
                group["weight_decay"] = 0.0
            groups.append(group)
    return self.optimizer_class(groups, **self.kwargs)

Optimizer.__call__ = ordered_optimizer

if sys.argv[1] != "--resume":
    config_path, run_name = sys.argv[1:3]
    run_id = f"{run_name}-{os.environ['SLURM_JOB_ID']}"
    # Physical batch 1 bounds the batch-dependent startup/training memory.
    # Accumulation averages four structures before each optimizer update.
    # The epoch ceiling comes from the YAML, including for long continuations.
    overrides = [
        "fitting/trainer_kwargs/accelerator=gpu",
        "fitting/trainer_kwargs/devices=1",
        "fitting/trainer_kwargs/precision=32-true",
        "fitting/trainer_kwargs/accumulate_grad_batches=4",
        "fitting/early_stopping/patience=300",
        "fitting/loader_kwargs/batch_size=1",
        "fitting/loader_kwargs/num_workers=0",
        "fitting/loader_kwargs/persistent_workers=false",
        "fitting/loader_kwargs/pin_memory=false",
        f"general/run_id={run_id}",
        f"wandb/name={run_id}",
    ]
    print("Config:", config_path, "Run:", run_id)
    print("Overrides:", overrides)
    raw = nested_merge_all(*(get_data_from_cli_arg(a) for a in [config_path, *overrides]))
    # Model, weighted MSE losses and learning rate come from the YAML.
    train_from_config(raw)
else:
    # Resume only a run CREATED WITH THIS SCRIPT. Older runs may not have
    # enough metadata to reconstruct the original optimizer parameter order.
    run_dir = Path(sys.argv[2]).resolve()
    checkpoint = run_dir / "checkpoints/last.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old_groups = state["optimizer_states"][0]["param_groups"]
    if any("param_names" not in group for group in old_groups):
        raise RuntimeError("Resume requires a checkpoint created with this script.")
    raw = yaml.safe_load((run_dir / "train-config.yaml").read_text())
    _, config = instantiate_config_from_dict(raw, TrainingConfig)
    configure_general_options(config.general.torch, config.general.seed)
    model = parse_model(config.model)
    data = config.get_data(model)
    data.train.prepare_data()
    data.valid.prepare_data()
    data.train.setup()
    data.valid.setup()
    # Supply eval_metrics, missing from graph-pes 1.0.2's resume command.
    task = TrainingTask.load_from_checkpoint(
        str(checkpoint), map_location="cpu",
        model=model, loss=parse_loss(config.loss),
        optimizer=config.fitting.get_optimizer(),
        scheduler=config.fitting.get_scheduler(),
        eval_metrics=get_all_eval_metrics([data.train, data.valid]),
    )
    check_optimizer = config.fitting.get_optimizer()(task.model)
    assert [g["param_names"] for g in check_optimizer.param_groups] == [
        g["param_names"] for g in old_groups
    ], "Optimizer parameter groups do not match the checkpoint"
    del check_optimizer, state
    log_to_file(run_dir)
    trainer = trainer_from_config(config, run_dir)
    kwargs = dict(config.fitting.loader_kwargs)
    kwargs.pop("shuffle", None)
    train_loader = GraphDataLoader(data.train, shuffle=True, **kwargs)
    valid_loader = GraphDataLoader(data.valid, shuffle=False, **kwargs)
    print("Resuming:", checkpoint)
    trainer.fit(task, train_loader, valid_loader, ckpt_path=str(checkpoint))
TRAIN_PY

