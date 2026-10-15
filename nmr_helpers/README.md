# NMR helpers

Run commands from the project root, with this directory beside `configs/` and `data/`.
Copy the entire `nmr_helpers/` directory to ARC together with the updated YAML files.

- `extract_ms_isotropy.py`: prepare `MS_Iso_QM = trace(ms)/3` labels, plus the scalar target and pre-fit compatibility adapters for graph-pes 1.0.2.
- `coordination_loss.py`: experiment 08's coordination-weighted training loss.
- `shielding_transfer.py`: experiments 09/11's energy/force backbone transfer (full or frozen).

The YAML files import these as `nmr_helpers.<module>.<object>`.
In notebooks, `import nmr_helpers` before loading trusted older checkpoints:
it registers their original module names for pickle compatibility.
Historical run configs remain unchanged. For a legacy config/launcher that imports
an old module name first, set `PYTHONPATH="$PWD:$PWD/nmr_helpers${PYTHONPATH:+:$PYTHONPATH}"`.

Label preparation (not needed when data are already prepared):

```bash
python -m nmr_helpers --help
python -m nmr_helpers data/train.xyz data/valid.xyz
```

Explicit input paths restrict preparation to those files. The no-argument default
includes raw, training, validation and test files. Existing outputs are never overwritten.

## Experiments 10 and 11

- **10**: random initialisation, all layers trainable, shielding MSE, LR `1e-4`.
- **11**: experiment-06 embeddings and all interaction layers copied and frozen;
  fresh scalar shielding readout and per-element shielding offsets/scales trainable,
  shielding MSE, LR `1e-4`. Offsets/scales are initialised from training labels only.
- Both use seed 42, the existing 800/50 train/validation split, physical batch 1
  with accumulation 4, 300 epochs, patience 300, and validation-selected checkpoints.
- Experiment 09 remains full fine-tuning because `freeze_backbone` defaults to false.

Copy the updated `shielding_transfer.py` with both configs to ARC before launching.
Experiment 11 also needs `graph-pes-results/Kavanagh-Med-9059603/model.pt`.
From the project root:

```bash
mkdir -p logs
sbatch train.sbatch configs/10-TensorNequIP-MSIsotropy-Scratch-LR1e-4.yaml TensorNequIP-NMR-Scratch-LR1e-4
sbatch train.sbatch configs/11-TensorNequIP-MSIsotropy-FrozenTransfer.yaml TensorNequIP-NMR-FrozenTransfer
```

Evaluate the best checkpoints on the same validation sites as 09, reporting
unweighted Si/O MAE and RMSE separately. No test split is configured.
