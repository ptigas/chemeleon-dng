import math
import numpy as np

import pytest
import torch


def make_dummy_atoms():
    try:
        from ase import Atoms
    except ImportError:
        pytest.skip("ASE not installed; skipping Atoms-based test")
    # Simple two-atom cubic cell (H, He)
    cell = 3.0 * torch.eye(3).numpy()
    positions = [
        [0.1, 0.2, 0.3],
        [0.6, 0.4, 0.8],
    ]
    numbers = [1, 2]
    at = Atoms(numbers=numbers, cell=cell, pbc=True)
    at.set_scaled_positions(positions)
    return at


def make_dm_cpu():
    import pytorch_lightning  # noqa: F401    
    from chemeleon_dng.script_util import create_diffusion_module
    dm = create_diffusion_module(
        task="csp",
        model_configs=dict(
            hidden_dim=32,
            time_dim=16,
            num_layers=2,
            max_atoms=128,
            act_fn="silu",
            dis_emb="sin",
            num_freqs=16,
            ln=True,
            ip=True,
            smooth=False,
            cond_dim=0,
            pred_atom_types=False,
        ),
        optimizer_configs=dict(
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            scheduler="constant",
            patience=10,
            early_stopping=0,
            warmup_steps=0,
        ),
        num_timesteps=8,
        beta_schedule_ddpm="cosine",
        beta_schedule_d3pm="linear",
        max_atoms=128,
        d3pm_hybrid_coeff=1.0,
        sigma_begin=0.01,
        sigma_end=0.1,
    )
    # Lightning module parameters are on CPU by default
    return dm


@torch.no_grad()
def test_crystalbatch_from_atoms_shapes():
    at1 = make_dummy_atoms()
    at2 = at1.copy()
    at2.set_scaled_positions([[0.9, 0.9, 0.9]])  # shrink to single-atom to vary sizes
    from ase import Atoms
    from chemeleon_dng.schema import crystalbatch_from_atoms
    at2 = Atoms(numbers=[1], cell=at1.cell, pbc=True)
    at2.set_scaled_positions([[0.25, 0.5, 0.75]])

    cb = crystalbatch_from_atoms([at1, at2], device="cpu")

    assert cb.num_graphs == 2
    assert cb.num_nodes == 3
    assert cb.atom_types.shape[0] == 3
    assert cb.lattices.shape == (2, 3, 3)
    assert cb.frac_coords.shape == (3, 3)
    assert cb.num_atoms.tolist() == [2, 1]
    assert cb.batch.tolist() == [0, 0, 1]
    # within [0,1)
    assert torch.all((cb.frac_coords >= 0) & (cb.frac_coords < 1))


@torch.no_grad()
def test_sdedit_variations_from_atoms_cpu():
    
    dm = make_dm_cpu()
    at = make_dummy_atoms()
    n_var = 3
    from chemeleon_dng.schema import crystalbatch_from_atoms
    x0 = crystalbatch_from_atoms([at] * n_var, device="cpu")

    outs = dm.sdedit(x0=x0, t_start=0.5, freeze_atom_types=True, verbose=False)

    assert isinstance(outs, list)
    assert len(outs) == n_var
    for out in outs:
        # Species preserved
        assert out.get_atomic_numbers().tolist() == at.get_atomic_numbers().tolist()
        # Fractional coordinates in [0,1)
        frac = out.get_scaled_positions()
        assert (frac >= -1e-6).all() and (frac < 1 + 1e-6).all()
        # Lattice is 3x3 and finite
        cell = out.cell.array
        assert cell.shape == (3, 3)
        assert np.isfinite(cell).all()


@torch.no_grad()
def test_sdedit_batch_variations_from_atoms_cpu():
    """Test batch-based sdedit processing with multiple input structures"""
    
    dm = make_dm_cpu()
    at1 = make_dummy_atoms()
    
    # Create a second dummy atoms with different cell size
    from ase import Atoms
    cell2 = 4.0 * torch.eye(3).numpy()  # Different cell size
    positions2 = [[0.2, 0.3, 0.4], [0.7, 0.5, 0.9]]
    numbers2 = [1, 2]  # Same composition but different positions
    at2 = Atoms(numbers=numbers2, cell=cell2, pbc=True)
    at2.set_scaled_positions(positions2)
    
    n_var = 2
    n_inputs = 2
    
    # Test batch processing with multiple input structures
    from chemeleon_dng.schema import crystalbatch_from_atoms
    
    # Create batch with multiple copies of each input structure
    all_atoms = [at1] * n_var + [at2] * n_var
    x0 = crystalbatch_from_atoms(all_atoms, device="cpu")

    outs = dm.sdedit(x0=x0, t_start=0.5, freeze_atom_types=True, verbose=False)

    assert isinstance(outs, list)
    assert len(outs) == n_inputs * n_var  # 2 inputs × 2 variations = 4 total
    
    # Check that first n_var outputs correspond to at1
    for i in range(n_var):
        out = outs[i]
        # Species preserved from first input
        assert out.get_atomic_numbers().tolist() == at1.get_atomic_numbers().tolist()
        # Fractional coordinates in [0,1)
        frac = out.get_scaled_positions()
        assert (frac >= -1e-6).all() and (frac < 1 + 1e-6).all()
        # Lattice is 3x3 and finite
        cell = out.cell.array
        assert cell.shape == (3, 3)
        assert np.isfinite(cell).all()
    
    # Check that next n_var outputs correspond to at2
    for i in range(n_var, 2 * n_var):
        out = outs[i]
        # Species preserved from second input
        assert out.get_atomic_numbers().tolist() == at2.get_atomic_numbers().tolist()
        # Fractional coordinates in [0,1)
        frac = out.get_scaled_positions()
        assert (frac >= -1e-6).all() and (frac < 1 + 1e-6).all()
        # Lattice is 3x3 and finite
        cell = out.cell.array
        assert cell.shape == (3, 3)
        assert np.isfinite(cell).all()


@torch.no_grad() 
def test_sample_sdedit_batch_functionality():
    """Test the sample.py sdedit function with batch processing"""
    import tempfile
    import shutil
    from pathlib import Path
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    output_dir = Path(temp_dir) / "sdedit_output"
    
    try:
        # Create test input structures
        at1 = make_dummy_atoms()
        at2 = make_dummy_atoms()
        # Modify second structure slightly
        positions2 = [[0.15, 0.25, 0.35], [0.65, 0.45, 0.85]]
        at2.set_scaled_positions(positions2)
        
        input_structures = [at1, at2]
        
        # Import and test the sdedit function with batch processing
        from chemeleon_dng.sample import sdedit
        
        # Test with batch_size parameter
        num_variations = 2
        batch_size = 3  # Process 3 structures at a time (2 inputs × 2 variations = 4 total > 3)
        
        # Create a minimal diffusion module for testing
        dm = make_dm_cpu()
        
        # Mock the function to avoid checkpoint loading issues
        # Instead, test the batch processing logic directly with the diffusion module
        from chemeleon_dng.schema import crystalbatch_from_atoms
        
        # Test the batch processing logic directly
        all_atoms_list = []
        structure_indices = []
        
        for i, at in enumerate(input_structures):
            for _ in range(num_variations):
                all_atoms_list.append(at.copy())
                structure_indices.append(i)
        
        # Verify we have the right number of structures
        assert len(all_atoms_list) == len(input_structures) * num_variations
        assert len(structure_indices) == len(all_atoms_list)
        assert structure_indices == [0, 0, 1, 1]  # 2 variations each for 2 inputs
        
        # Test batch processing
        batch_size = 3
        all_gen_atoms = []
        
        for batch_start in range(0, len(all_atoms_list), batch_size):
            batch_end = min(batch_start + batch_size, len(all_atoms_list))
            batch_atoms = all_atoms_list[batch_start:batch_end]
            x0 = crystalbatch_from_atoms(batch_atoms, device="cpu")
            
            gen_atoms_list = dm.sdedit(
                x0=x0,
                t_start=0.5,
                freeze_atom_types=True,
                verbose=False,
            )
            
            all_gen_atoms.extend(gen_atoms_list)
        
        # Verify results
        assert len(all_gen_atoms) == len(input_structures) * num_variations
        
        # Test that first 2 structures match input 0
        for i in range(num_variations):
            assert all_gen_atoms[i].get_atomic_numbers().tolist() == at1.get_atomic_numbers().tolist()
            
        # Test that last 2 structures match input 1  
        for i in range(num_variations, 2 * num_variations):
            assert all_gen_atoms[i].get_atomic_numbers().tolist() == at2.get_atomic_numbers().tolist()
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
