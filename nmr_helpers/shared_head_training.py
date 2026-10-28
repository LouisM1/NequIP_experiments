"""Minimal continuation and balanced-loss support for experiments 13A/13B."""
from copy import deepcopy
from pathlib import Path
import math
import shutil

import torch
from pytorch_lightning import Callback
from graph_pes.training.loss import Loss


def from_shielding_checkpoint(model, checkpoint):
    """Load trusted scalar shielding weights without changing the freeze mask."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = {k.removeprefix("model."): v for k, v in state["state_dict"].items()
               if k.startswith("model.")}
    model.load_state_dict(weights, strict=True)
    if not model._has_been_pre_fit.item():
        raise ValueError("Expected a pre-fitted shielding checkpoint")
    return model


class ElementBalancedMSE(Loss):
    """Equal Si/O mean squared errors divided by fixed training variances.

    The same objective is used in training and validation. Both species must
    occur in each batch, as in this dataset's 48 Si / 96 O structures.
    """
    def __init__(self, si_std, o_std, weight=1000.0):
        super().__init__(weight=weight, is_per_atom=True)
        if not all(math.isfinite(s) and s > 0 for s in (si_std, o_std)):
            raise ValueError("Training standard deviations must be positive")
        self.si_std, self.o_std = float(si_std), float(o_std)

    @property
    def required_properties(self):
        return ["tensor"]

    @property
    def name(self):
        return "element_balanced_standardized_mse"

    def forward(self, model, graph, predictions):
        pred, target = predictions["tensor"], graph.properties["tensor"]
        if pred.shape != target.shape or target.shape != graph.Z.shape:
            raise ValueError("Expected one scalar shielding per atom")
        si, oxygen = graph.Z == 14, graph.Z == 8
        if not bool(si.any() and oxygen.any() and (si | oxygen).all()):
            raise ValueError("Expected a batch containing only Si and O, both present")
        error2 = (pred - target).square()
        return 0.5 * (error2[si].mean() / self.si_std**2
                      + error2[oxygen].mean() / self.o_std**2)


class ElementRMSELogger(Callback):
    """Log exact epoch RMSE for each species, without a second model forward."""
    def _reset(self, stage):
        setattr(self, "_" + stage, None)

    def _update(self, stage, task, batch):
        # Observe predictions passed into the loss, without another forward.
        def collect(module, args):
            _, graph, predictions = args
            if "tensor" not in predictions:
                return
            error2 = (predictions["tensor"].detach() - batch.properties["tensor"]).double().square()
            values = torch.stack([v for z in (14, 8) for v in
                                  (error2[batch.Z == z].sum(), (batch.Z == z).sum())])
            previous = getattr(self, "_" + stage)
            setattr(self, "_" + stage, values if previous is None else previous + values)
        self._hook = task.total_loss.register_forward_pre_hook(collect)

    def _remove(self):
        self._hook.remove()

    def _log(self, stage, trainer, task):
        values = getattr(self, "_" + stage)
        if values is None or trainer.sanity_checking:
            return
        for i, element in enumerate(("Si", "O")):
            task.log(f"{stage}/metrics/{element}_rmse", (values[2*i] / values[2*i+1]).sqrt(),
                     on_step=False, on_epoch=True)

    def on_train_epoch_start(self, trainer, pl_module): self._reset("train")
    def on_validation_epoch_start(self, trainer, pl_module): self._reset("valid")
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): self._update("train", pl_module, batch)
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0): self._update("valid", pl_module, batch)
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): self._remove()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0): self._remove()
    def on_train_epoch_end(self, trainer, pl_module): self._log("train", trainer, pl_module)
    def on_validation_epoch_end(self, trainer, pl_module): self._log("valid", trainer, pl_module)


class _CheckOptimizerOrder(Callback):
    def __init__(self, groups):
        self.names = [g["param_names"] for g in groups]

    def on_fit_start(self, trainer, pl_module):
        actual = [g.get("param_names") for g in trainer.optimizers[0].param_groups]
        if actual != self.names:
            raise RuntimeError("Optimizer parameter order differs from source checkpoint")


def prepare_continuation(trainer, output_dir, checkpoint, model):
    """Copy/rebase checkpoint paths, preserving source results and fit state."""
    checkpoint, output_dir = Path(checkpoint).resolve(), Path(output_dir).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    groups = state["optimizer_states"][0]["param_groups"]
    if any("param_names" not in group for group in groups):
        raise ValueError("Full continuation needs saved optimizer parameter names")
    if trainer.max_epochs <= state["epoch"] + 1:
        raise ValueError("max_epochs must exceed the completed checkpoint epochs")
    destination = output_dir / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    # Keep the historical best available even if continuation never improves it.
    best_source = checkpoint.parent / "best.ckpt"
    shutil.copy2(best_source, destination / "best.ckpt")
    # Preserve the historical best export too, if no new epoch improves it.
    from graph_pes.training.callbacks import SaveBestModel
    baseline = from_shielding_checkpoint(deepcopy(model), best_source)
    torch.save(baseline.cpu(), output_dir / "model.pt")
    for callback in state["callbacks"].values():
        if callback.get("monitor") == "valid/loss/total":
            for cb in trainer.callbacks:
                if isinstance(cb, SaveBestModel):
                    cb.best_val_loss = float(callback["best_model_score"])
    for key, callback in state["callbacks"].items():
        if not key.startswith("ModelCheckpoint"):
            continue
        callback["dirpath"] = str(destination)
        for field in ("best_model_path", "kth_best_model_path", "last_model_path"):
            if callback.get(field):
                callback[field] = str(destination / Path(callback[field]).name)
        callback["best_k_models"] = {
            str(destination / Path(p).name): score
            for p, score in callback.get("best_k_models", {}).items()
        }
    resume_path = output_dir / "continuation.ckpt"
    torch.save(state, resume_path)
    trainer.callbacks.append(_CheckOptimizerOrder(groups))
    trainer.ckpt_path = str(resume_path)


def train_from_config(raw):
    """Use the normal graph-pes pipeline, optionally restoring full fit state.

    `continuation_checkpoint` is a launcher extension, stripped before normal
    config parsing. We leave architecture, optimization and data handling to
    graph-pes. Existing --resume runs use their own latest checkpoint instead.
    """
    import graph_pes.scripts.train as training
    from nmr_helpers.training_fixes import install_training_fixes
    install_training_fixes()
    raw = deepcopy(raw)
    checkpoint = raw.pop("continuation_checkpoint", None)
    if checkpoint is None:
        return training.train_from_config(raw)
    original = training.trainer_from_config

    def factory(config, output_dir):
        from graph_pes.config.shared import parse_model
        trainer = original(config, output_dir)
        prepare_continuation(trainer, output_dir, checkpoint, parse_model(config.model))
        return trainer

    training.trainer_from_config = factory
    try:
        return training.train_from_config(raw)
    finally:
        training.trainer_from_config = original
