"""NMR data preparation, scalar model compatibility, losses and transfer.

Register the original module names so trusted experiment 07/08 checkpoints
remain loadable after importing nmr_helpers. Importing does not prepare data.
"""
import sys
from . import extract_ms_isotropy, coordination_loss, shielding_transfer

for _module in (extract_ms_isotropy, coordination_loss, shielding_transfer):
    sys.modules.setdefault(_module.__name__.rsplit(".", 1)[-1], _module)
del _module
