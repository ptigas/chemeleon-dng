from __future__ import annotations
from collections import OrderedDict
from pydantic import BaseModel, Field, ConfigDict

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor


if TYPE_CHECKING:  # Only for type hints; avoid runtime dependency when ASE is missing
    from ase import Atoms  # pragma: no cover


class CrystalBatch(BaseModel):
    """
    A schema for a batch of crystal structures.
    """

    # Pydantic v2
    model_config = ConfigDict(arbitrary_types_allowed=True)

    atom_types: Tensor
    lattices: Tensor
    frac_coords: Tensor
    num_atoms: Tensor
    batch: Tensor  # batch_idx
    num_graphs: Optional[int] = None
    num_nodes: Optional[int] = None
    noise_atom_types: Optional[Tensor] = None
    noise_lattices: Optional[Tensor] = None
    noise_frac_coords: Optional[Tensor] = None


class Trajectory(BaseModel):
    """
    A schema for a trajectory of crystal structures.
    Each crystal structure is represented as a CrystalBatch.
    """

    # Pydantic v2
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    total_steps: int
    container: OrderedDict[int, CrystalBatch] = Field(default_factory=OrderedDict)

    def __getitem__(self, t: int):
        return self.container[t]

    def __setitem__(self, t: int, value: CrystalBatch) -> None:
        self.container[t] = value

    def __len__(self) -> int:
        return len(self.container)

    def get_atoms(self, t: int = 0, idx: Optional[int] = None) -> "Atoms" | list["Atoms"]:
        from ase.build.tools import sort  # local import to avoid hard dependency at module import time
        from ase import Atoms  # local import to avoid hard dependency at module import time
        trajectory_step = self.container[t]

        # If atom type is greater than 100 + 1, set it to 0
        trajectory_step.atom_types = torch.where(
            trajectory_step.atom_types <= 101, trajectory_step.atom_types, 0
        )
        split_atom_types = torch.split(
            trajectory_step.atom_types, trajectory_step.num_atoms.tolist()
        )
        split_frac_coords = torch.split(
            trajectory_step.frac_coords, trajectory_step.num_atoms.tolist()
        )

        atoms_list = []
        for i, (frac_coords, atom_types) in enumerate(
            zip(split_frac_coords, split_atom_types)
        ):
            atoms = Atoms(
                numbers=atom_types.detach().cpu().numpy(),
                cell=trajectory_step.lattices[i].detach().cpu().numpy(),
                pbc=True,
            )
            positions = frac_coords.detach().cpu().numpy()
            atoms.set_scaled_positions(positions)
            atoms_list.append(sort(atoms))
        if idx is None:
            return atoms_list
        else:
            return atoms_list[idx]

    def get_trajectory(self, idx: Optional[int] = None):
        if idx is None:
            return [self.get_atoms(t, None) for t in range(self.total_steps + 1)]
        else:
            return [self.get_atoms(t, idx) for t in range(self.total_steps + 1)]


def crystalbatch_from_atoms(
    atoms: "Atoms" | list["Atoms"], device: torch.device | str
) -> CrystalBatch:
    """
    Construct a CrystalBatch from a single ASE Atoms or a list of Atoms.
    """
    device = torch.device(device)
    atoms_list = atoms if isinstance(atoms, list) else [atoms]

    num_graphs = len(atoms_list)
    numbers_list: list[int] = []
    num_atoms_list: list[int] = []
    lattices_list: list[Tensor] = []
    frac_coords_list: list[Tensor] = []

    for i, at in enumerate(atoms_list):
        nums = at.get_atomic_numbers().tolist()
        numbers_list.extend(nums)
        num_atoms_list.append(len(nums))
        lattices_list.append(torch.tensor(at.cell.array, dtype=torch.float32))
        frac_coords_list.append(
            torch.tensor(at.get_scaled_positions(wrap=True), dtype=torch.float32)
        )

    num_nodes = sum(num_atoms_list)
    batch_idx = torch.tensor(
        [i for i, n in enumerate(num_atoms_list) for _ in range(n)], dtype=torch.long
    )

    return CrystalBatch(
        atom_types=torch.tensor(numbers_list, dtype=torch.long, device=device),
        lattices=torch.stack(lattices_list, dim=0).to(device),
        frac_coords=torch.vstack(frac_coords_list).to(device) % 1.0,
        num_atoms=torch.tensor(num_atoms_list, dtype=torch.long, device=device),
        batch=batch_idx.to(device),
        num_graphs=num_graphs,
        num_nodes=num_nodes,
    )
