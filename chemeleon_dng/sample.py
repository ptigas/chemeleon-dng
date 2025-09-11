"""
This script samples crystal structures using a trained diffusion model.
It supports three types of tasks:
- CSP (Crystal Structure Prediction): Predicts stable crystal structures from given atom types
- DNG (De Novo Generation): Generates new crystal structures from scratch
- GUIDE (Guiding Generation): Generates structures with guidance
"""

import json
from pathlib import Path
import fire
import numpy as np
from pymatgen.core import Composition, Structure
from monty.serialization import dumpfn

import torch
from chemeleon_dng.diffusion.diffusion_module import DiffusionModule
from chemeleon_dng.schema import crystalbatch_from_atoms
from chemeleon_dng.dataset.num_atom_distributions import NUM_ATOM_DISTRIBUTIONS
from chemeleon_dng.download_util import get_checkpoint_path
from ase import Atoms

DEFAULT_MODEL_PATH = {
    "csp": "ckpts/chemeleon_csp_alex_mp_20_v0.0.2.ckpt",
    "dng": "ckpts/chemeleon_dng_alex_mp_20_v0.0.2.ckpt",
    "guide": ".",
}


def sample_csp(
    dm: DiffusionModule,
    formulas: str | tuple,
    num_samples: int,
    batch_size: int,
    output_path: Path,
):
    # Parse formulas
    if isinstance(formulas, (tuple, list)):
        formula_list = [f.strip() for f in formulas]
    else:
        formula_list = [formulas.strip()]
    print(f"Generating {num_samples} samples for each formula: {formula_list}")

    # Generate samples in batches
    def repeat_elements(l, times):
        return [x for x in l for _ in range(times)]

    total_formula_list = repeat_elements(formula_list, num_samples)
    for i in range(0, len(total_formula_list), batch_size):
        print(f"Generating batch #{i // batch_size + 1} with {batch_size} samples.")
        batch_formula_list = total_formula_list[i : i + batch_size]

        atom_types = []
        num_atoms = []
        for formula in batch_formula_list:
            comp = Composition(formula)
            atomic_numbers = [el.Z for el, amt in comp.items() for _ in range(int(amt))]
            atom_types.extend(atomic_numbers)
            num_atoms.append(len(atomic_numbers))

        assert len(atom_types) == sum(num_atoms)
        gen_atoms_list = dm.sample(
            task="csp", atom_types=atom_types, num_atoms=num_atoms
        )

        # Save generated structures
        for j, atoms in enumerate(gen_atoms_list):
            atoms.write(output_path / f"sample_{i+j}_{atoms.get_chemical_formula()}.cif")  # type: ignore


def sample_dng(
    dm: DiffusionModule,
    num_atom_distribution: str | list,
    num_samples: int,
    batch_size: int,
    output_path: Path,
):
    if isinstance(num_atom_distribution, str):
        atom_distribution = NUM_ATOM_DISTRIBUTIONS[num_atom_distribution]
        num_atoms = np.random.choice(
            list(atom_distribution.keys()),
            p=list(atom_distribution.values()),
            size=num_samples,
        ).tolist()
    elif isinstance(num_atom_distribution, list):
        num_atoms = num_atom_distribution
        assert (
            len(num_atoms) == num_samples
        ), "num_atom_distribution must be a list of length num_samples."
    else:
        raise ValueError(
            "num_atom_distribution must be either a string or a list of integers."
        )
    print(
        f"Generating {num_samples} samples with atom distributions: {num_atom_distribution}"
    )

    # Generate samples in batches
    for i in range(0, num_samples, batch_size):
        print(f"Generating batch #{i // batch_size + 1} with {batch_size} samples.")
        batch_num_atoms = num_atoms[i : i + batch_size]
        gen_atoms_list = dm.sample(task="dng", num_atoms=batch_num_atoms)

        # Save generated structures
        for j, atoms in enumerate(gen_atoms_list):
            atoms.write(output_path / f"sample_{i+j}_{atoms.get_chemical_formula()}.cif")  # type: ignore


def sample(
    task: str,
    num_samples: int = 100,
    batch_size: int | None = None,
    formulas: str | tuple | list | None = None,  # Only for CSP task
    num_atom_distribution: str | list | None = "mp-20",  # Only for DNG task
    model_path: str | None = None,
    output_dir: str = "./results",
    device: str | None = None,
    save_json: bool = True,
):
    """Sample crystal structures using a trained diffusion model.

    :param task: Type of sampling task ('csp', 'dng', or 'guide').
                 'csp' for Crystal Structure Prediction,
                 'dng' for De Novo Generation,
                 'guide' for Guided Generation.
    :param num_samples: Number of samples to generate per formula (for CSP) or total samples (for DNG), defaults to 100
    :param batch_size: Number of samples to generate per batch, defaults to num_samples if None
    :param formulas: Chemical formula(s) for crystal structure prediction (CSP task).
                    Can be a single string like 'NaCl' or a tuple of strings like ('NaCl', 'SiO2').
                    Required for CSP task, defaults to None
    :param num_atom_distribution: Distribution of number of atoms per unit cell for DNG task.
                                 Can be a predefined distribution name (e.g., 'mp-20') or
                                 a list of integers specifying the number of atoms for each sample.
                                 Required for DNG task, defaults to "mp-20"
    :param model_path: Path to the checkpoint file of the trained model.
                     If None, uses the default checkpoint for the specified task, defaults to None
    :param output_dir: Directory to save the generated structures, defaults to "./results"
    :param device: Device to run the model on ('cuda' or 'cpu').
                  If None, uses CUDA if available, otherwise CPU, defaults to None
    :param save_json: Whether to save the generated structures in JSON format, defaults to True

    Examples\n
    --------\n
    # CSP: Generate 100 samples each for NaCl, OCl3, and UO15 formulas, processing 150 samples at a time
    >>> sample(task="csp", formulas="NaCl, OCl3, UO15", num_samples=100, batch_size=150, output_dir="./results")
    or
    >>> sample(task="csp", formulas=["NaCl", "OCl3", "UO15"], num_samples=100, batch_size=150, output_dir="./results")

    # CSP: Generate 100 samples for a single formula at a time
    >>> sample(task="csp", formulas="SiO2", num_samples=100)

    # DNG: Generate 100 samples with specific atom counts [7,8]
    >>> sample(task="dng", num_atom_distribution=[7, 8] * 50, num_samples=100)

    # DNG: Generate 100 samples using the 'mp-20' atom distribution
    >>> sample(task="dng", num_samples=100)

    # Using a custom checkpoint path
    >>> sample(task="csp", formulas="NaCl", model_path="path/to/custom/checkpoint.ckpt")

    # Explicitly specify device
    >>> sample(task="dng", device="cpu", num_samples=5)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set default batch size if not provided
    if batch_size is None:
        batch_size = num_samples

    # Set default checkpoint path if not provided
    if model_path is None:
        model_path = get_checkpoint_path(task, DEFAULT_MODEL_PATH)
    print(f"Using checkpoint path: {model_path}")

    # Load the diffusion module
    dm = DiffusionModule.load_from_checkpoint(model_path, map_location=device)

    # Validate the checkpoint’s task matches requested task
    assert (
        dm.model.task.name.lower() == task.lower()  # type: ignore
    ), f"Checkpoint task does not match the provided task '{task}'."

    # Set Output Directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"The generated structures will be saved in: {output_path}")
    existing_files = list(output_path.glob("sample_*.cif"))
    if existing_files:
        print(
            f"Warning: {len(existing_files)} existing CIF files found in the output directory."
            " Recommend emptying the directory before sampling to avoid overwriting."
        )

    # Validate task
    task = task.lower()
    if task == "csp":
        assert formulas is not None, "Formulas must be provided for CSP task."
        if isinstance(formulas, str):
            if formulas.endswith(".json"):
                with open(formulas, "r") as f:
                    formulas = json.load(f)
            else:
                formulas = tuple(formulas.split(","))
        sample_csp(
            dm=dm,
            formulas=formulas,
            num_samples=num_samples,
            batch_size=batch_size,
            output_path=output_path,
        )
    elif task == "dng":
        assert (
            num_atom_distribution is not None
        ), "num_atom_distribution must be provided for DNG task."
        sample_dng(
            dm=dm,
            num_atom_distribution=num_atom_distribution,
            num_samples=num_samples,
            batch_size=batch_size,
            output_path=output_path,
        )
    elif task == "guide":
        assert (
            dm.model.task.name.lower() == task  # type: ignore
        ), "Checkpoint task does not match the provided task."
    else:
        raise ValueError(
            f"Unsupported task: {task}. Supported tasks are: csp, dng, guide."
        )

    # Save generated structures in JSON format
    if save_json:
        gen_atoms_files = list(output_path.glob("sample_*.cif"))
        all_gen_atoms_list = [Structure.from_file(file) for file in gen_atoms_files]
        dumpfn(all_gen_atoms_list, output_path / "generated_structures.json.gz")
        print(
            f"The {len(all_gen_atoms_list)} generated structures saved in JSON format at: {output_path / 'generated_structures.json.gz'}"
        )


def sdedit(
    input_path: str | Path | Atoms | list,
    t_start: int | float = 0.6,
    num_variations: int = 4,
    model_path: str | None = None,
    output_dir: str = "./results_sdedit",
    device: str | None = None,
    freeze_atom_types: bool = True,
    save_json: bool = True,
    batch_size: int | None = None,
):
    """SDEdit-style variations starting from an existing structure.

    - ``input_path``: CIF file, directory of CIFs, an ASE ``Atoms`` or list of ``Atoms``.
    - ``t_start``: integer timestep in [1, T] or ratio (0, 1].
    - ``num_variations``: number of variations per input structure.
    - ``freeze_atom_types``: keep atom types fixed to preserve composition.
    - ``batch_size``: maximum number of structures to process in a single batch. If None, processes all structures together.
    """
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model
    if model_path is None:
        # Default to CSP checkpoint to preserve atom types by design
        model_path = get_checkpoint_path("csp", DEFAULT_MODEL_PATH)
    print(f"Using checkpoint path: {model_path}")
    dm = DiffusionModule.load_from_checkpoint(model_path, map_location=device)

    # Resolve inputs: files or ASE Atoms
    inputs_atoms: list[Atoms] = []
    input_names: list[str] = []

    if isinstance(input_path, Atoms):
        inputs_atoms = [input_path]
        input_names = [input_path.get_chemical_formula()]  # type: ignore
    elif isinstance(input_path, list) and input_path and isinstance(input_path[0], Atoms):
        inputs_atoms = input_path  # type: ignore
        input_names = [at.get_chemical_formula() for at in inputs_atoms]  # type: ignore
    else:
        p = Path(input_path)  # type: ignore[arg-type]
        if p.is_dir():
            files = sorted(list(p.glob("*.cif")))
            assert files, f"No CIF files found in directory: {p}"
        else:
            files = [p]
        for fi in files:
            s = Structure.from_file(str(fi))
            at = s.to_ase_atoms()
            inputs_atoms.append(at)
            input_names.append(fi.stem)
    print(f"Found {len(inputs_atoms)} input structure(s)")

    # Output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set default batch size if not provided
    if batch_size is None:
        batch_size = len(inputs_atoms) * num_variations
    
    # Create all structure variations for batch processing
    all_atoms_list = []
    structure_indices = []  # Track which input structure each variation belongs to
    
    for i, at in enumerate(inputs_atoms):
        for _ in range(num_variations):
            all_atoms_list.append(at.copy())
            structure_indices.append(i)
    
    print(f"Processing {len(all_atoms_list)} total variations ({len(inputs_atoms)} structures × {num_variations} variations each) in batches of {batch_size}")
    
    # Process in batches
    all_gen_atoms = []
    for batch_start in range(0, len(all_atoms_list), batch_size):
        batch_end = min(batch_start + batch_size, len(all_atoms_list))
        print(f"Processing batch {batch_start//batch_size + 1}: structures {batch_start} to {batch_end-1}")
        
        batch_atoms = all_atoms_list[batch_start:batch_end]
        x0 = crystalbatch_from_atoms(batch_atoms, device=device)

        gen_atoms_list = dm.sdedit(
            x0=x0,
            t_start=t_start,
            freeze_atom_types=freeze_atom_types,
            verbose=False,
        )
        
        all_gen_atoms.extend(gen_atoms_list)
    
    # Save results grouped by input structure
    for i, (gen_atoms, struct_idx) in enumerate(zip(all_gen_atoms, structure_indices)):
        input_name = input_names[struct_idx]
        variation_idx = i - struct_idx * num_variations  # Calculate variation index for this structure
        gen_atoms.write(out_dir / f"{input_name}_sdedit_{variation_idx}.cif")  # type: ignore

    # Save JSON bundle
    if save_json:
        gen_atoms_files = list(out_dir.glob("*_sdedit_*.cif"))
        all_gen_atoms_list = [Structure.from_file(file) for file in gen_atoms_files]
        dumpfn(all_gen_atoms_list, out_dir / "sdedit_structures.json.gz")
        print(
            f"Saved {len(all_gen_atoms_list)} SDEdit structures to: {out_dir / 'sdedit_structures.json.gz'}"
        )


def main():
    """Main CLI entry point for chemeleon-sample command."""
    fire.Fire(sample)


if __name__ == "__main__":
    main()
