LOCAL EXPERIMENT TEMPLATE

1. Duplicate this folder and rename it for your experiment.
2. From the new folder run: bash setup-local.sh
   This recreates .venv at its new location using uv's package cache.
   Do not rely on a copied venv: its scripts retain absolute paths.
3. Open experiment.ipynb and select the new folder's .venv/bin/python.
4. Write the hypothesis, edit train.yaml, and run the small local check.
5. Copy your updated train.yaml to the matching experiment on ARC.
   Copy the notebook too if you want it available there.
6. Submit the job on ARC, then download results for local analysis.

The matching ARC template contains the GPU job and identical datasets.
The job reads train.yaml; it does not execute experiment.ipynb.
Custom training logic must be part of the code the job actually executes.
The supplied model is a SchNet starter, not a pretrained model or a
ready-made NequIP transfer-learning configuration.

DATA
train.xyz: 800 fitting structures; valid.xyz: 50 validation structures.
test.xyz: 150 held-out structures; raw.xyz: original 850 source structures.
All are under data/. split-indices.json records source indices/checksum.
Keep the split fixed across comparisons. Only train/valid are read by default.
The energy offset in train.yaml belongs to this training subset; revisit it
when changing the dataset or composition.
