# NMR helpers

Run commands from the project root, with this directory beside `configs/` and `data/`.
Copy the entire `nmr_helpers/` directory to ARC together with the updated YAML files.

- `extract_ms_isotropy.py`: prepare `MS_Iso_QM = trace(ms)/3` labels, plus the scalar target and pre-fit compatibility adapters for graph-pes 1.0.2.
- `coordination_loss.py`: experiment 08's coordination-weighted training loss.
- `shielding_transfer.py`: experiment 09's energy/force backbone transfer.

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
