"""Process-local fixes for graph-pes 1.0.2 checkpoint compatibility."""
import warnings

import torch
from pytorch_lightning.callbacks import EarlyStopping
from graph_pes.training.callbacks import EarlyStoppingWithLogging


def _early_stopping_state(self):
    # Keep the original class/state key so existing checkpoints still load.
    return {**EarlyStopping.state_dict(self), **self.state}


def _load_early_stopping_state(self, state):
    for key in self.state:
        if key in state:
            self.state[key] = state[key]
    if 'wait_count' in state and 'best_score' in state:
        EarlyStopping.load_state_dict(self, state)
    else:
        # graph-pes 1.0.2 saved only these logging counters. With its default
        # monitor and min_delta=0 they track the same validation decisions.
        if self.monitor != 'valid/loss/total' or self._min_delta != 0:
            raise ValueError('Legacy early-stopping state can only be recovered '
                             'for valid/loss/total with min_delta=0.')
        required = {'total_checks', 'best_val_loss_check', 'best_val_loss'}
        if not required.issubset(state):
            raise ValueError('Checkpoint is missing early-stopping history.')
        self.wait_count = int(state['total_checks'] - state['best_val_loss_check'])
        self.best_score = torch.as_tensor(state['best_val_loss']).clone()
        self.stopped_epoch = 0
        warnings.warn('Recovered legacy early-stopping patience and best score '
                      'from graph-pes logging counters.', stacklevel=2)


def install_training_fixes():
    """Preserve both Lightning's stopping state and graph-pes logging state."""
    EarlyStoppingWithLogging.state_dict = _early_stopping_state
    EarlyStoppingWithLogging.load_state_dict = _load_early_stopping_state
