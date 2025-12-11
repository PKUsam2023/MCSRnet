import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
import hydra
import pytorch_lightning as pl
from tqdm import tqdm
from xtcsg.common.utils import PROJECT_ROOT
from xtcsg.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from xtcsg.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()  # auto-save args/kwargs
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model  # use model sub-config

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )  # create optimizer from config
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )  # create scheduler if enabled
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


class SinusoidalTimeEmbeddings(nn.Module):
    """Sinusoidal time embedding (Transformer style)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)  # freq scale
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]  # time × frequencies
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # sin|cos
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            latent_dim=self.hparams.latent_dim + self.hparams.time_dim,
            _recursive_=False,
        )  # decoder network

        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)  # diffusion schedules

        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)  # time encoder

        self.keep_lattice = self.hparams.cost_lattice < 1e-5  # skip lattice loss?
        self.keep_coords = self.hparams.cost_coord < 1e-5  # skip coord loss?

    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)  # sample time steps
        time_emb = self.time_embedding(times)  # encode time
        xrd_embeddings = batch.xrd_embeddings.view(batch_size, -1)  # flatten XRD features

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]  # diffusion α, β

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1.0 - alphas_cumprod)  # noise mixing coefficients

        c0_per_atom = c0.repeat_interleave(batch.num_atoms)[:, None]
        c1_per_atom = c1.repeat_interleave(batch.num_atoms)[:, None]

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)  # lattice matrix
        frac_coords = batch.frac_coords
        frac_occupancy = batch.frac_occupancy

        frac_occupancy = torch.clamp((frac_occupancy * 100).round(), 1, 100) - 1
        frac_occupancy = frac_occupancy.long()  # discretize occupancy

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)  # lattice/coord noise

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l  # noised lattice
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]

        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0  # noised coords

        frac_occupancy_onehot = F.one_hot(frac_occupancy, num_classes=MAX_ATOMIC_NUM).float()
        rand_oc = torch.randn_like(frac_occupancy_onehot)
        input_frac_occupancy = c0_per_atom * frac_occupancy_onehot + c1_per_atom * rand_oc  # noised occupancy

        if self.keep_coords:
            input_frac_coords = frac_coords  # optionally freeze coords

        if self.keep_lattice:
            input_lattice = lattices  # optionally freeze lattice

        pred_l, pred_x, pred_oc = self.decoder(
            xrd_embeddings,
            time_emb,
            batch.atom_types,
            input_frac_coords,
            input_frac_occupancy,
            input_lattice,
            batch.num_atoms,
            batch.batch,
        )  # model prediction

        tar_x = (
            d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom)
            / torch.sqrt(sigmas_norm_per_atom)
        )  # wrapped normal target

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)
        loss_occupancy = F.mse_loss(pred_oc, rand_oc)  # component losses

        loss = (
            self.hparams.cost_lattice * loss_lattice
            + self.hparams.cost_coord * loss_coord
            + self.hparams.cost_occupancy * loss_occupancy
        )  # weighted total loss

        return {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
            "loss_occupancy": loss_occupancy,
        }

    @torch.no_grad()


    def sample(self, batch, step_lr = 1e-5):
        """Reverse diffusion sampling: generate crystal structures."""

        batch_size = batch.num_graphs

        # Initial noise states at T
        l_T = torch.randn([batch_size, 3, 3]).to(self.device)
        x_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        oc_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device)

        time_start = self.beta_scheduler.timesteps
        xrd_embeddings = batch.xrd_embeddings.view(batch_size, -1)

        # Optional: use ground-truth coords/lattice
        if self.keep_coords:
            x_T = batch.frac_coords
        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        # Store initial state
        traj = {time_start: {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'frac_coords': x_T % 1.,
            'frac_occupancy': oc_T,
            'lattices': l_T
        }}

        # Reverse-time loop
        for t in tqdm(range(time_start, 0, -1)):
            times = torch.full((batch_size,), t, device=self.device)
            time_emb = self.time_embedding(times)

            # Scheduler parameters
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            # Reverse SDE coefficients
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            # Current states
            x_t = traj[t]['frac_coords']
            oc_t = traj[t]['frac_occupancy']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T

            # -------- Corrector step (PC sampler) --------
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_oc = torch.randn_like(oc_T) if t > 1 else torch.zeros_like(oc_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x, pred_oc = self.decoder(
                xrd_embeddings, time_emb, batch.atom_types,
                x_t, oc_t, l_t, batch.num_atoms, batch.batch
            )
            pred_x = pred_x * torch.sqrt(sigma_norm)

            # Corrector update
            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x \
                        if not self.keep_coords else x_t
            l_t_minus_05 = l_t if self.keep_lattice else l_t
            oc_t_minus_05 = oc_t if self.keep_lattice else oc_t

            # -------- Predictor step --------
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            rand_oc = torch.randn_like(oc_T) if t > 1 else torch.zeros_like(oc_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = torch.sqrt((adjacent_sigma_x**2 *
                            (sigma_x**2 - adjacent_sigma_x**2)) /
                            (sigma_x**2))

            pred_l, pred_x, pred_oc = self.decoder(
                xrd_embeddings, time_emb, batch.atom_types,
                x_t_minus_05, oc_t_minus_05, l_t_minus_05,
                batch.num_atoms, batch.batch
            )
            pred_x = pred_x * torch.sqrt(sigma_norm)

            # Predictor update
            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x \
                        if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l \
                        if not self.keep_lattice else l_t

            oc_t_minus_1 = c0 * (oc_t_minus_05 - c1 * pred_oc) + sigmas * rand_oc \
                        if not self.keep_lattice else oc_t

            # Save trajectory
            traj[t - 1] = {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': x_t_minus_1 % 1.,
                'frac_occupancy': oc_t_minus_1,
                'lattices': l_t_minus_1
            }

        # Stack entire trajectory
        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'all_frac_coords': torch.stack([traj[i]['frac_coords']
                                            for i in range(time_start, -1, -1)]),
            'all_frac_occupancy': torch.stack([traj[i]['frac_occupancy']
                                            for i in range(time_start, -1, -1)]),
            'all_lattices': torch.stack([traj[i]['lattices']
                                        for i in range(time_start, -1, -1)]),
            'low_atomic_numbers': batch.low_atomic_numbers
        }

        # Final step: convert occupancy to discrete classes
        traj[0]['low_atomic_numbers'] = batch.low_atomic_numbers
        traj[0]["frac_occupancy"] = (torch.argmax(traj[0]["frac_occupancy"], dim=-1) + 1) / 100

        return traj[0], traj_stack

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """One training iteration: forward pass, loss logging."""
        
        output_dict = self(batch)

        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_occupancy = output_dict['loss_occupancy']
        loss = output_dict['loss']

        # Log individual and total losses
        self.log_dict(
            {
                'train_loss': loss,
                'lattice_loss': loss_lattice,
                'coord_loss': loss_coord,
                'occupancy_loss': loss_occupancy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Prevent NaN from breaking training
        if loss.isnan():
            return None

        return loss


    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation iteration."""
        
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        # Log validation metrics
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Test iteration."""
        
        output_dict = self(batch)
        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        # Log test metrics
        self.log_dict(log_dict)
        return loss


    def compute_stats(self, output_dict, prefix):
        """Format metrics for logging."""
        
        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_occupancy = output_dict['loss_occupancy']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_occupancy_loss': loss_occupancy,
        }

        return log_dict, loss
