import enum
import copy
from tqdm import tqdm
from typing import Optional

import torch
from torch import Tensor

from chemeleon_dng.base_module import BaseModule
from chemeleon_dng.cspnet import CSPNet
from chemeleon_dng.schema import CrystalBatch, Trajectory
from chemeleon_dng.script_util import create_model
from chemeleon_dng.diffusion.models import D3PM, DDPM, DSM
from chemeleon_dng.diffusion.diffusion_scheduler import uniform_sample_t


class DiffusionModuleTask(enum.Enum):
    """
    Which task of the model.
    """

    CSP = enum.auto()  # Crystal Structure Prediction
    DNG = enum.auto()  # De Novo Generation
    GUIDE = enum.auto()  # Guiding Generation (Only available for DNG)


class DiffusionType(enum.Enum):
    """
    Type of diffusion model.
    """

    D3PM = enum.auto()  # Discrete Denoising Diffusion Probabilistic Models (Discrete)
    DDPM = enum.auto()  # Denoising Diffusion Probabilistic Model (Variance-preserving)
    DSM = enum.auto()  # Denising Score Matching (Variance-exploding)


class DiffusionModule(BaseModule):
    """
    Utilities for training and sampling diffusion models.
    """

    def __init__(
        self,
        *,
        task: DiffusionModuleTask,
        num_timesteps: int,
        diffusion_atom_type: D3PM,
        diffusion_lattice: DDPM,
        diffusion_frac_coord: DSM,
        model_configs: dict,
        optimizer_configs: dict,
    ):
        super().__init__(**optimizer_configs)
        self.save_hyperparameters()

        if task == DiffusionModuleTask.CSP:
            assert (
                diffusion_atom_type is None
            ), "Diffusion atom type should be None, when CSP"
        self.model = create_model(**model_configs, task=task.name)
        self.num_timesteps = num_timesteps
        self.diffusion_atom_type = diffusion_atom_type
        self.diffusion_lattice = diffusion_lattice
        self.diffusion_frac_coord = diffusion_frac_coord

    def q_sample(
        self,
        x_start: CrystalBatch,
        t: Tensor,
        noise_atom_types: Optional[Tensor] = None,
        noise_lattices: Optional[Tensor] = None,
        noise_frac_coords: Optional[Tensor] = None,
    ) -> CrystalBatch:
        """
        Sample from the forward process q(x_t | x_0).
        """
        return CrystalBatch(
            atom_types=(
                self.diffusion_atom_type.q_sample(
                    x_start=x_start.atom_types,
                    t=t,
                    batch_idx=x_start.batch,
                    noise=noise_atom_types,
                )
                if self.diffusion_atom_type is not None
                else x_start.atom_types
            ),
            lattices=self.diffusion_lattice.q_sample(
                x_start=x_start.lattices,
                t=t,
                noise=noise_lattices,
            ),
            frac_coords=self.diffusion_frac_coord.q_sample(
                x_start=x_start.frac_coords,
                t=t,
                batch_idx=x_start.batch,
                noise=noise_frac_coords,
            ),
            num_atoms=x_start.num_atoms,
            batch=x_start.batch,
        )

    def calculate_loss(
        self,
        x_start: CrystalBatch,
        t: Optional[Tensor] = None,
        cond_embeds: Optional[Tensor] = None,
    ):
        if t is None:
            t = uniform_sample_t(
                num_timesteps=self.num_timesteps, batch_size=x_start.num_graphs
            ).to(self.device)
        # Create noise for atom types, lattices, and fractional coordinates
        if self.diffusion_atom_type is not None:
            noise_atom_types = torch.rand(
                len(x_start.atom_types), self.diffusion_atom_type.max_atoms
            ).to(x_start.atom_types.device)
        else:
            noise_atom_types = None
        noise_lattices = torch.randn_like(x_start.lattices)
        noise_frac_coords = torch.randn_like(x_start.frac_coords)
        # Sample from the forward process q(x_t | x_0)
        x_t = self.q_sample(
            x_start=x_start,
            t=t,
            noise_atom_types=noise_atom_types,
            noise_lattices=noise_lattices,
            noise_frac_coords=noise_frac_coords,
        )
        # Model prediction
        model_output = _model_prediction(
            model=self.model,
            x_t=x_t,
            t=t,
            cond_embeds=cond_embeds,
        )
        # Calculate the loss
        res = {}
        if self.diffusion_atom_type is not None:
            loss_atom_types = self.diffusion_atom_type.training_loss(
                pred_x_start_logits=model_output.atom_types,
                x_start=x_start.atom_types,
                x_t=x_t.atom_types,
                t=t,
                batch_idx=x_t.batch,
            )
        else:
            loss_atom_types = {}
        loss_lattices = self.diffusion_lattice.training_loss(
            model_output=model_output.lattices,
            noise=noise_lattices,
        )
        loss_frac_coords = self.diffusion_frac_coord.training_loss(
            model_output=model_output.frac_coords,
            noise=noise_frac_coords,
            t=t,
            batch_idx=x_t.batch,
        )
        res.update({f"{k}_atom_types": v for k, v in loss_atom_types.items()})
        res.update({f"{k}_lattices": v for k, v in loss_lattices.items()})
        res.update({f"{k}_frac_coords": v for k, v in loss_frac_coords.items()})
        total_loss = (
            (loss_atom_types["loss"] + loss_lattices["loss"] + loss_frac_coords["loss"])
            if self.diffusion_atom_type is not None
            else loss_lattices["loss"] + loss_frac_coords["loss"]
        )
        res.update({"total_loss": total_loss})
        return res

    def p_sample(
        self,
        x_t: CrystalBatch,
        t: Tensor,
        step_lr: float = 1e-5,
        cond_scale: float = 2.0,
        cond_embeds: Optional[Tensor] = None,
        null_cond_embeds: Optional[Tensor] = None,
        freeze_atom_types: bool = False,
    ) -> CrystalBatch:
        """
        Sample from the reverse process p(x_{t-1} | x_t).
        """
        # Predictor
        model_predictor_output = _model_prediction(
            model=self.model,
            x_t=x_t,
            t=t,
            cond_scale=cond_scale,
            cond_embeds=cond_embeds,
            null_cond_embeds=null_cond_embeds,
        )
        predictor_output = CrystalBatch(
            atom_types=(
                x_t.atom_types
                if (freeze_atom_types or self.diffusion_atom_type is None)
                else self.diffusion_atom_type.p_sample(
                    pred_x_start_logits=model_predictor_output.atom_types,
                    x_t=x_t.atom_types,
                    t=t,
                    batch_idx=x_t.batch,
                )
            ),
            lattices=self.diffusion_lattice.p_sample(
                model_output=model_predictor_output.lattices,
                x_t=x_t.lattices,
                t=t,
            ),
            frac_coords=self.diffusion_frac_coord.p_sample(
                model_output=model_predictor_output.frac_coords,
                x_t=x_t.frac_coords,
                t=t,
                batch_idx=x_t.batch,
                mode="predictor",
            ),
            num_atoms=x_t.num_atoms,
            batch=x_t.batch,
        )

        # Corrector (update only frac_coords)
        model_corrector_output = _model_prediction(
            model=self.model,
            x_t=predictor_output,
            t=t,
            cond_scale=cond_scale,
            cond_embeds=cond_embeds,
            null_cond_embeds=null_cond_embeds,
        )
        return CrystalBatch(
            atom_types=predictor_output.atom_types,
            lattices=predictor_output.lattices,
            frac_coords=self.diffusion_frac_coord.p_sample(
                model_output=model_corrector_output.frac_coords,
                x_t=predictor_output.frac_coords,
                t=t,
                batch_idx=x_t.batch,
                mode="corrector",
                step_lr=step_lr,
            ),
            num_atoms=x_t.num_atoms,
            batch=x_t.batch,
            num_graphs=x_t.num_graphs,
            num_nodes=x_t.num_nodes,
        )

    def sample(
        self,
        task: str,
        num_atoms: list[int] | Tensor,
        atom_types: Optional[list[int] | Tensor] = None,
        return_trajectory: bool = False,
        step_lr: float = 1e-5,
        verbose: bool = True,
    ):
        if task.lower() == "csp":
            assert self.model.task.name.lower() == "csp", "Model task should be CSP"  # type: ignore
            if atom_types is not None:
                assert len(atom_types) == sum(
                    num_atoms
                ), "Length of atom_types and num_atoms should be the same."
            else:
                raise ValueError("atom_types must be provided for CSP task.")
        elif task.lower() == "dng":
            assert self.model.task.name.lower() == "dng", "Model task should be DNG"  # type: ignore
            if atom_types is not None:
                atom_types = None
                print("The given atom_type in arguments will be ignored for DNG task")
        elif task.lower() == "guide":
            assert self.model.task.name.lower() == "guide", "Model task should be GUIDE"  # type: ignore
        else:
            raise ValueError(
                f"Unknown task {task}. Supported tasks are 'csp', 'dng', and 'guide'."
            )
        assert num_atoms is not None

        # Construct an initial batch of crystal structures
        num_graphs = len(num_atoms)
        num_nodes = sum(num_atoms)
        x_t = CrystalBatch(
            atom_types=(
                torch.as_tensor(atom_types, dtype=torch.long, device=self.device)
                if atom_types is not None
                else torch.zeros(
                    num_nodes,
                    dtype=torch.long,
                    device=self.device,
                )
            ),
            lattices=torch.randn((num_graphs, 3, 3), device=self.device),
            frac_coords=torch.rand((num_nodes, 3), device=self.device) % 1.0,
            num_atoms=torch.as_tensor(num_atoms, device=self.device),
            batch=torch.tensor(
                [i for i, n in enumerate(num_atoms) for _ in range(n)],
                device=self.device,
            ),
            num_graphs=num_graphs,
            num_nodes=num_nodes,
        )

        # Create Trajectory object to store the generated structures
        trajectory = Trajectory(total_steps=self.num_timesteps)
        trajectory.container[self.num_timesteps] = x_t

        # Sample from the reverse process p(x_{t-1} | x_t)
        timesteps = range(self.num_timesteps, 0, -1)
        if verbose:
            timesteps = tqdm(timesteps, desc="Sampling")

        for timestep in timesteps:
            t = (
                torch.ones((len(num_atoms),), dtype=torch.long, device=self.device)
                * timestep
            )
            with torch.no_grad():
                x_t = self.p_sample(x_t=x_t, t=t, step_lr=step_lr)
            trajectory.container[timestep - 1] = x_t

        if return_trajectory:
            return trajectory
        return trajectory.get_atoms(t=0)

    def sdedit(
        self,
        *,
        x0: CrystalBatch,
        t_start: int | float,
        return_trajectory: bool = False,
        step_lr: float = 1e-5,
        cond_scale: float = 2.0,
        cond_embeds: Optional[Tensor] = None,
        null_cond_embeds: Optional[Tensor] = None,
        verbose: bool = True,
        freeze_atom_types: bool = True,
    ):
        """
        SDEdit-style sampling: start from an existing crystal x0, add noise to a
        mid/high timestep t* to obtain x_{t*}, then run the reverse process to t=0.

        - If ``t_start`` is a float in (0, 1], it is treated as a fraction of the
          total diffusion steps and converted to an integer timestep.
        - If ``freeze_atom_types`` is True, atom types are kept fixed during
          both noising and denoising (preserves composition/identity).
        """
        assert x0.atom_types.device == self.device, "x0 must be on the same device"

        if isinstance(t_start, float):
            assert 0.0 < t_start <= 1.0, "t_start as ratio must be in (0, 1]"
            t_idx = int(torch.ceil(torch.tensor(t_start * self.num_timesteps)).item())
        else:
            t_idx = int(t_start)
        t_idx = max(1, min(self.num_timesteps, t_idx))

        # Prepare noise per component
        noise_atom_types = None
        if (self.diffusion_atom_type is not None) and (not freeze_atom_types):
            noise_atom_types = torch.rand(
                len(x0.atom_types), self.diffusion_atom_type.max_atoms, device=self.device
            )
        noise_lattices = torch.randn_like(x0.lattices)
        noise_frac_coords = torch.randn_like(x0.frac_coords)

        # Build x_{t*}
        t = torch.ones((x0.num_graphs,), dtype=torch.long, device=self.device) * t_idx
        if freeze_atom_types or (self.diffusion_atom_type is None):
            x_t = CrystalBatch(
                atom_types=x0.atom_types,
                lattices=self.diffusion_lattice.q_sample(
                    x_start=x0.lattices, t=t, noise=noise_lattices
                ),
                frac_coords=self.diffusion_frac_coord.q_sample(
                    x_start=x0.frac_coords, t=t, batch_idx=x0.batch, noise=noise_frac_coords
                ),
                num_atoms=x0.num_atoms,
                batch=x0.batch,
                num_graphs=x0.num_graphs,
                num_nodes=x0.num_nodes,
            )
        else:
            x_t = self.q_sample(
                x_start=x0,
                t=t,
                noise_atom_types=noise_atom_types,
                noise_lattices=noise_lattices,
                noise_frac_coords=noise_frac_coords,
            )

        # Trajectory
        trajectory = Trajectory(total_steps=t_idx)
        trajectory.container[t_idx] = x_t

        # Reverse process from t_idx -> 0
        timesteps = range(t_idx, 0, -1)
        if verbose:
            timesteps = tqdm(timesteps, desc="SDEdit Sampling")
        for timestep in timesteps:
            t_curr = torch.ones((x0.num_graphs,), dtype=torch.long, device=self.device) * timestep
            with torch.no_grad():
                x_t = self.p_sample(
                    x_t=x_t,
                    t=t_curr,
                    step_lr=step_lr,
                    cond_scale=cond_scale,
                    cond_embeds=cond_embeds,
                    null_cond_embeds=null_cond_embeds,
                    freeze_atom_types=freeze_atom_types,
                )
            trajectory.container[timestep - 1] = x_t

        if return_trajectory:
            return trajectory
        return trajectory.get_atoms(t=0)


def _model_prediction(
    model: CSPNet,
    x_t: CrystalBatch,
    t: Tensor,
    cond_embeds: Optional[Tensor] = None,
    null_cond_embeds: Optional[Tensor] = None,
    cond_scale: float = 1.0,
) -> CrystalBatch:
    if cond_embeds is not None:
        assert null_cond_embeds is not None
        cond_model_output = model(
            atom_types=x_t.atom_types,
            lattices=x_t.lattices,
            frac_coords=x_t.frac_coords,
            num_atoms=x_t.num_atoms,
            batch_idx=x_t.batch,
            t=t,
            cond_embeds=cond_embeds,
        )
        null_model_output = model(
            atom_types=x_t.atom_types,
            lattices=x_t.lattices,
            frac_coords=x_t.frac_coords,
            num_atoms=x_t.num_atoms,
            batch_idx=x_t.batch,
            t=t,
            cond_embeds=null_cond_embeds,
        )
        output = copy.deepcopy(cond_model_output)
        output.atom_types = (
            1 - cond_scale
        ) * null_model_output.atom_types + cond_scale * cond_model_output.atom_types
        output.lattices = (
            1 - cond_scale
        ) * null_model_output.lattices + cond_scale * cond_model_output.lattices
        output.frac_coords = (
            (1 - cond_scale) * null_model_output.frac_coords
            + cond_scale * cond_model_output.frac_coords
        )
    else:
        output = model(
            atom_types=x_t.atom_types,
            lattices=x_t.lattices,
            frac_coords=x_t.frac_coords,
            num_atoms=x_t.num_atoms,
            batch_idx=x_t.batch,
            t=t,
        )
    return output
