"""Prepare scalar NMR labels and support their graph-pes 1.0.2 pre-fitting.

From the experiment directory:
    .venv/bin/python extract_ms_isotropy.py
    .venv/bin/python -m graph_pes.scripts.train configs/07-TensorNequIP-MSIsotropy.yaml

Defaults: read data/{raw,train,valid,test}.xyz and write separate copies to
 data/nmr_iso/ with exactly one added array, MS_Iso_QM (ppm). Existing splits,
frame/atom order, tensors, energies, forces and metadata are retained. No
referencing or sign change is applied: the target is trace(ms)/3.

The YAML imports ScalarTensorAdditionModel from this file, so keep this file
beside the configs/ and data/ directories, including when copying to ARC.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import iread, write
from graph_pes.models import TensorAdditionModel


class ScalarTarget(str):
    """The string '0e', also iterable as irreps for graph-pes 1.0.2 pruning.

    TensorNequIP's pruned constructor iterates the target as irreps, while
    its readout expects a string without a multiplicity prefix. This adapter
    satisfies both without changing pruning or the resulting architecture.
    """

    def __new__(cls):
        return super().__new__(cls, "0e")

    def __iter__(self):
        from e3nn.o3 import Irreps
        return iter(Irreps(str(self)))


class ScalarTensorAdditionModel(TensorAdditionModel):
    """TensorAdditionModel with scalar-compatible pre-fitting for graph-pes 1.0.2.

    The upstream tensor pre-fit assumes multi-column targets, and its ridge
    helper broadcasts incorrectly when scikit-learn squeezes a single target.
    Compute the same one-hot ridge means/variances analytically for C=1.
    Forward passes, learnable parameters, loss shapes and architecture are
    unchanged. Only training graphs are used to initialise offsets/scales.
    """

    def pre_fit_all_components(self, graphs):
        import torch
        from graph_pes.atomic_graph import to_batch
        from graph_pes.models import LearnableTensorOffset
        from graph_pes.models.components.scaling import LocalTensorScaler
        from graph_pes.utils.nn import PerElementParameter

        if self._has_been_pre_fit.item():
            return
        batch = to_batch(list(graphs))
        values = batch.properties["tensor"]
        if values.shape != (len(batch.Z),):
            raise ValueError("Expected one scalar MS_Iso_QM value per atom")
        elements = torch.unique(batch.Z)
        with torch.no_grad():
            means, scales = {}, {}
            for z in elements.tolist():
                labels = values[batch.Z == z]
                # Equivalent to Ridge(fit_intercept=False, alpha=1e-5)
                # on a one-hot element design matrix, as used upstream.
                denominator = labels.numel() + 1e-5
                mean = labels.sum() / denominator
                variance = ((labels - mean) ** 2).sum() / denominator
                means[z] = mean
                scales[z] = variance.clamp_min(0.001).sqrt()
            for module in self.modules():
                if isinstance(module, LearnableTensorOffset):
                    for z in elements.tolist():
                        if not torch.any(module._pre_specified_Zs == z):
                            module._offsets[z] = means[z]
                elif isinstance(module, LocalTensorScaler):
                    for z in elements.tolist():
                        module.per_element_scaling[z] = scales[z]
                elif hasattr(module, "pre_fit"):
                    module.pre_fit(batch)
            for parameter in self.parameters():
                if isinstance(parameter, PerElementParameter):
                    parameter.register_elements(elements.tolist())
            for module in self.modules():
                if hasattr(module, "_has_been_pre_fit"):
                    module._has_been_pre_fit.fill_(1)


def extract_file(source: Path, destination: Path) -> tuple[int, int]:
    """Write a new labelled file, refusing to replace any existing file."""
    from soprano.properties.nmr import MSIsotropy

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must be different files")
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = atoms = 0
    # Exclusive creation protects originals and previous outputs alike.
    with destination.open("x") as output:
        try:
            for index, frame in enumerate(iread(source, index=":", format="extxyz")):
                if "MS_Iso_QM" in frame.arrays or "MS_Iso_QM" in frame.info:
                    raise ValueError(f"{source}, frame {index}: MS_Iso_QM already exists")
                if "ms" not in frame.arrays:
                    raise ValueError(f"{source}, frame {index}: missing ms tensors")
                tensors = np.asarray(frame.arrays["ms"])
                if tensors.shape not in ((len(frame), 9), (len(frame), 3, 3)):
                    raise ValueError(f"{source}, frame {index}: invalid ms shape {tensors.shape}")
                # Prevent Soprano from adding ms_shielding and ms_isotropy.
                values = MSIsotropy.get(frame, tag="ms", save_array=False)
                if values.shape != (len(frame),) or not np.isfinite(values).all():
                    raise ValueError(f"{source}, frame {index}: invalid isotropic shieldings")
                frame.set_array("MS_Iso_QM", values)
                write(output, frame, format="extxyz")
                frames += 1
                atoms += len(frame)
            if not frames:
                raise ValueError(f"{source}: no frames found")
        except BaseException:
            # Remove only this newly created incomplete output.
            destination.unlink()
            raise
    return frames, atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path,
                        default=[Path("data") / f"{name}.xyz" for name in ("raw", "train", "valid", "test")])
    parser.add_argument("--output-dir", type=Path, default=Path("data/nmr_iso"))
    args = parser.parse_args()
    outputs = [args.output_dir / source.name for source in args.inputs]
    if len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("Input basenames must be unique")
    # Check all paths before generating any files; there is no overwrite mode.
    for source, destination in zip(args.inputs, outputs):
        if not source.is_file():
            parser.error(f"Input does not exist: {source}")
        if destination.exists():
            parser.error(f"Output already exists: {destination}")
        if destination.resolve() in {p.resolve() for p in args.inputs}:
            parser.error(f"Output would overwrite an input: {destination}")
    for source, destination in zip(args.inputs, outputs):
        frames, atoms = extract_file(source, destination)
        print(f"{source} -> {destination}: {frames} frames, {atoms} atoms; added MS_Iso_QM (ppm)")


if __name__ == "__main__":
    main()
