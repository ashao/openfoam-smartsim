import numpy as np
import physicsnemo.sym
import physicsnemo.sym.loss.aggregator
import torch
import torch.optim as optim

from sympy import Symbol, Function, Rational
from sympy import sqrt as sym_sqrt
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.activation import Activation

from abc import ABC, abstractmethod

# Override unnecessary cuda dependencies
import torch.cuda.nvtx

torch.cuda.nvtx.range_push = lambda *_, **__: None
torch.cuda.nvtx.range_pop = lambda *_, **__: None

from random import shuffle

default_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def MLP(layer_size=20, nr_layers=3, **kwargs):
    return FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z")],
        output_keys=[Key("u"), Key("v"), Key("w")],
        layer_size=layer_size,
        nr_layers=nr_layers,
        activation_fn=Activation.TANH,
    )


class MeshMotionBulk(ABC, PDE):

    def __init__(self):
        self.dim = 3
        # Position variables
        self.x, self.y, self.z = Symbol("x"), Symbol("y"), Symbol("z")

        # Displacements as a function of x
        self.u = Function("u")(self.x, self.y, self.z)
        self.v = Function("v")(self.x, self.y, self.z)
        self.w = Function("w")(self.x, self.y, self.z)

    @abstractmethod
    def _define_equations(self):
        pass


class Laplace3d(MeshMotionBulk):
    def __init__(self):
        super().__init__()
        self._define_equations()

    def _define_equations(self):
        x, y, z = self.x, self.y, self.z
        u, v, w = self.u, self.v, self.w

        laplace_u = u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)
        laplace_v = v.diff(x, 2) + v.diff(y, 2) + v.diff(z, 2)
        laplace_w = w.diff(x, 2) + w.diff(y, 2) + w.diff(z, 2)

        self.equations = {
            "mom_x": laplace_u,
            "mom_y": laplace_v,
            "mom_z": laplace_w,
        }


class NavierCauchy3d(MeshMotionBulk):
    def __init__(self, nu=0.01):
        super().__init__()
        self.nu = nu
        self._define_equations(nu)

    def _define_equations(self, nu):
        x, y, z = self.x, self.y, self.z
        u, v, w = self.u, self.v, self.w

        # Define terms of PDEs
        div_u = u.diff(x) + v.diff(y) + w.diff(z)

        laplace_u = u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)
        laplace_v = v.diff(x, 2) + v.diff(y, 2) + v.diff(z, 2)
        laplace_w = w.diff(x, 2) + w.diff(y, 2) + w.diff(z, 2)

        self.equations = {
            "mom_x": nu * laplace_u + div_u.diff(x),
            "mom_y": nu * laplace_v + div_u.diff(y),
            "mom_z": nu * laplace_w + div_u.diff(z),
        }

class StrainRate(MeshMotionBulk):
    def __init__(self):
        super().__init__()
        self._define_equations()

    def name(self):
        return "StrainTensor"

    def _define_equations(self):
        element = lambda u_i, u_j, x_i, x_j: Rational(1, 2) * (
            u_i.diff(x_j) + u_j.diff(x_i)
        )
        x, y, z = self.x, self.y, self.z
        u, v, w = self.u, self.v, self.w

        # Diagonal terms first
        e_xx = u.diff(x)
        e_yy = v.diff(y)
        e_zz = w.diff(z)

        # Off diagonal
        e_xy = element(u, v, x, y)
        e_xz = element(u, w, x, z)
        e_yz = element(v, w, y, z)

        # Euclidean norm of stress tensor (squared)
        self.equations = {
            "strain_norm": sym_sqrt(
                e_xx**2 + e_yy**2 + e_zz**2 + 2 * (e_xy**2 + e_xz**2 + e_yz**2)
            )
        }


class StrainRateNoShear(MeshMotionBulk):
    def __init__(self):
        super().__init__()
        self._define_equations()

    def name(self):
        return "StrainTensor"

    def _define_equations(self):
        element = lambda u_i, u_j, x_i, x_j: Rational(1, 2) * (
            u_i.diff(x_j) + u_j.diff(x_i)
        )
        x, y, z = self.x, self.y, self.z
        u, v, w = self.u, self.v, self.w

        # Off diagonal
        e_xy = element(u, v, x, y)
        e_xz = element(u, w, x, z)
        e_yz = element(v, w, y, z)

        # Euclidean norm of stress tensor (squared)
        self.equations = {"strain_norm": sym_sqrt(2 * (e_xy**2 + e_xz**2 + e_yz**2))}


class PINNTrainer(ABC):
    def __init__(
        self,
        model,
        boundary_points,
        bulk_points,
        bulk_equations,
        n_bulk_samples=500,
        lr=1e-2,
        loss_stop=0.5,
        device=default_device,
    ):
        self.eq = bulk_equations
        self.model = model.to(device)
        self.device = device
        self.physics_informer = PhysicsInformer(
            required_outputs=self.eq.equations.keys(),
            equations=self.eq,
            device=device,
            grad_method="autodiff",
        )

        # Store the initial boundary points
        self.boundary_points = torch.from_numpy(boundary_points).float().to(device)
        self.n_boundary_points = len(self.boundary_points)

        # Randomly sample bulk
        self.n_bulk_samples = n_bulk_samples
        self._sample_bulk(n_bulk_samples, bulk_points)

        # Batch the boundary and bulk points together
        self._batch_boundary_and_interior()

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_stop = loss_stop
        self._create_aggregator()

    def set_boundary_displacements(self, displacements):
        self.boundary_displacements = (
            torch.from_numpy(displacements).float().to(self.device)
        )

    def _sample_bulk(self, n_bulk_samples, bulk_points):
        indices = list(range(len(bulk_points)))
        shuffle(indices)
        train_indices = indices[:n_bulk_samples]
        self.bulk_points = (
            torch.from_numpy(bulk_points[train_indices]).float().to(self.device)
        )

        validation_indices = indices[n_bulk_samples:2*n_bulk_samples]
        self.bulk_validation_points = (
            torch.from_numpy(bulk_points[validation_indices])
            .float()
            .to(self.device)
        )

    def _batch_boundary_and_interior(self):
        self.X = torch.vstack([self.boundary_points, self.bulk_points, self.bulk_validation_points])
        self.X.requires_grad_(True)

        offsets = [
            self.n_boundary_points,
            self.n_boundary_points + self.n_bulk_samples
        ]

        self.boundary_indices = slice(0, offsets[0])
        self.bulk_train_indices = slice(offsets[0], offsets[1])
        self.bulk_validation_indices = slice(offsets[1], None)

    def _create_aggregator(self):
        nlosses = len(self.eq.equations) + 1
        weights = {
            k: torch.tensor(1.0) for k in self.eq.equations.keys()
        }
        weights["boundary"] = torch.tensor(10.0)
        self.agg_training = physicsnemo.sym.loss.aggregator.ResNorm(
            self.model.parameters(), nlosses, weights=weights
        )

    def _calc_residuals(self, y_pred):
        physics_informer_input = {
            "coordinates": self.X,
            "u": y_pred[:, 0],
            "v": y_pred[:, 1],
            "w": y_pred[:, 2],
        }
        residuals = self.physics_informer.forward(physics_informer_input)
        return residuals

    def _residual_to_loss(self, residuals):
        return torch.mean(residuals**2)

    def _boundary_loss(self, y_pred):
        return torch.mean(
            (y_pred-self.boundary_displacements)**2
        )

    def _model_forward(self, X):
        return self.model._impl.forward(X)

    def _calc_all_losses(self):
        y_pred = self._model_forward(self.X)
        residuals = self._calc_residuals(y_pred)
        train_losses = {
            k: self._residual_to_loss(v[self.bulk_train_indices]) for k, v in residuals.items()
        }
        train_losses["boundary"] = self._boundary_loss(y_pred[self.boundary_indices, :])
        validation_losses = {
            k: self._residual_to_loss(v[self.bulk_validation_indices]) for k, v in residuals.items()
        }

        return train_losses, validation_losses

    def step(self, iteration):
        self.optimizer.zero_grad()
        train_losses, validation_losses = self._calc_all_losses()
        agg_training_loss = self.agg_training.forward(train_losses, iteration)
        agg_training_loss.backward()
        self.optimizer.step()

        validation_losses ={
            "pde": np.mean([v.item() for k, v in validation_losses.items() if k != "boundary"]),
            "boundary": train_losses["boundary"].item()
        }

        return agg_training_loss, train_losses, validation_losses

    def step_bc_only(self):
        self.optimizer.zero_grad()
        y_pred = self._model_forward(self.boundary_points)
        loss = self._boundary_loss(y_pred)
        loss.backward()
        self.optimizer.step()
        return loss

    def reset(self):
        self.optimizer.state.clear()
