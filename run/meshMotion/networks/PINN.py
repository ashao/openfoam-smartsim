
import numpy as np
import physicsnemo.sym
import physicsnemo.sym.loss.aggregator
import torch
import torch.optim as optim

from sympy import Symbol, Function, Rational
from sympy import sqrt as sym_sqrt
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseConstraint
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.activation import Activation

from abc import ABC, abstractmethod

# Override unnecessary cuda dependencies
import torch.cuda.nvtx
torch.cuda.nvtx.range_push = lambda *_, **__: None
torch.cuda.nvtx.range_pop = lambda *_, **__: None

from random import shuffle


class MLP:
   def __init__(self, layer_size=20, nr_layers=3, **kwargs):
      self.model = FullyConnectedArch(
         input_keys=[Key("x"), Key("y"), Key("z")],
         output_keys=[Key("u"), Key("v"), Key("w")],
         layer_size=10,
         nr_layers=3,
         activation_fn=Activation.ELU
      )
   def __call__(self, X):
      return self.model._impl.forward(X)


class MeshMotionBulk(ABC, PDE):

   def __init__(self):
      # Position variables
      self.x, self.y, self.z = Symbol("x"), Symbol("y"), Symbol("z")

      # Displacements as a function of x
      self.u = Function("u")(self.x, self.y, self.z)
      self.v = Function("v")(self.x, self.y, self.z)
      self.w = Function("w")(self.x, self.y, self.z)

   @property
   @abstractmethod
   def name(self):
      pass

   @abstractmethod
   def _define_equations(self):
      pass

   @abstractmethod
   def create_interior_condition(self, nodes, bulk_points):
      pass


class Laplace3d(MeshMotionBulk):
   def __init__(self):
      super().__init__()
      self._define_equations()

   def name(self):
      return "Laplace"

   def _define_equations(self):
      x, y, z = self.x, self.y, self.z
      u, v, w = self.u, self.v, self.w

      laplace_u = u.diff(x,2) + u.diff(y,2) + u.diff(z,2)
      laplace_v = v.diff(x,2) + v.diff(y,2) + v.diff(z,2)
      laplace_w = w.diff(x,2) + w.diff(y,2) + w.diff(z,2)

      self.equations = {
         "mom_x": laplace_u,
         "mom_y": laplace_v,
         "mom_z": laplace_w,
      }

   def create_interior_condition(self, nodes, bulk_points):
      # Enforce the Navier-Lame equations in the interior
      n_bulk_points = bulk_points.shape[0]
      interior_condition = PointwiseConstraint.from_numpy(
         nodes=nodes,
         invar={
            "x": np.expand_dims(bulk_points[:,0], -1),
            "y": np.expand_dims(bulk_points[:,1], -1),
            "z": np.expand_dims(bulk_points[:,2], -1),
         },
         outvar={
            "mom_x": np.zeros((n_bulk_points,1)),
            "mom_y": np.zeros((n_bulk_points,1)),
            "mom_z": np.zeros((n_bulk_points,1)),
         },
         batch_size=n_bulk_points
      )
      return interior_condition

class NavierCauchy3d(MeshMotionBulk):
   def __init__(self, nu=0.025):
      super().__init__()
      self.nu = nu
      self._define_equations(nu)

   def name(self):
      return "NavierCauchy"

   def _define_equations(self, nu):
      x, y, z = self.x, self.y, self.z
      u, v, w = self.u, self.v, self.w

      # Define terms of PDEs
      div_u = u.diff(x) + v.diff(y) + w.diff(z)

      laplace_u = u.diff(x,2) + u.diff(y,2) + u.diff(z,2)
      laplace_v = v.diff(x,2) + v.diff(y,2) + v.diff(z,2)
      laplace_w = w.diff(x,2) + w.diff(y,2) + w.diff(z,2)

      self.equations = {
         "mom_x": nu*laplace_u + div_u.diff(x),
         "mom_y": nu*laplace_v + div_u.diff(y),
         "mom_z": nu*laplace_w + div_u.diff(z),
      }

   def create_interior_condition(self, nodes, bulk_points):
      # Enforce the Navier-Lame equations in the interior
      n_bulk_points = bulk_points.shape[0]
      interior_condition = PointwiseConstraint.from_numpy(
         nodes=nodes,
         invar={
            "x": np.expand_dims(bulk_points[:,0], -1),
            "y": np.expand_dims(bulk_points[:,1], -1),
            "z": np.expand_dims(bulk_points[:,2], -1),
         },
         outvar={
            "mom_x": np.zeros((n_bulk_points,1)),
            "mom_y": np.zeros((n_bulk_points,1)),
            "mom_z": np.zeros((n_bulk_points,1)),
         },
         batch_size=n_bulk_points
      )
      return interior_condition


class StrainRate(MeshMotionBulk):
   def __init__(self):
      super().__init__()
      self._define_equations()

   def name(self):
      return "StrainTensor"

   def _define_equations(self):
      element = lambda u_i, u_j, x_i, x_j: Rational(1,2)*(u_i.diff(x_j) + u_j.diff(x_i))
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
         "strain_norm": sym_sqrt(e_xx**2 + e_yy**2 + e_zz**2 + 2*(e_xy**2 + e_xz**2 + e_yz**2))
      }

   def create_interior_condition(self, nodes, bulk_points):
      # Enforce the Navier-Lame equations in the interior
      n_bulk_points = bulk_points.shape[0]
      interior_condition = PointwiseConstraint.from_numpy(
         nodes=nodes,
         invar={
            "x": np.expand_dims(bulk_points[:,0], -1),
            "y": np.expand_dims(bulk_points[:,1], -1),
            "z": np.expand_dims(bulk_points[:,2], -1),
         },
         outvar={
            "strain_norm": np.zeros((n_bulk_points,1)),
         },
         batch_size=n_bulk_points
      )
      return interior_condition


class StrainRateNoShear(MeshMotionBulk):
   def __init__(self):
      super().__init__()
      self._define_equations()

   def name(self):
      return "StrainTensor"

   def _define_equations(self):
      element = lambda u_i, u_j, x_i, x_j: Rational(1,2)*(u_i.diff(x_j) + u_j.diff(x_i))
      x, y, z = self.x, self.y, self.z
      u, v, w = self.u, self.v, self.w

      # Off diagonal
      e_xy = element(u, v, x, y)
      e_xz = element(u, w, x, z)
      e_yz = element(v, w, y, z)

      # Euclidean norm of stress tensor (squared)
      self.equations = {
         "strain_norm": sym_sqrt(2*(e_xy**2 + e_xz**2 + e_yz**2))
      }

   def create_interior_condition(self, nodes, bulk_points):
      # Enforce the Navier-Lame equations in the interior
      n_bulk_points = bulk_points.shape[0]
      interior_condition = PointwiseConstraint.from_numpy(
         nodes=nodes,
         invar={
            "x": np.expand_dims(bulk_points[:,0], -1),
            "y": np.expand_dims(bulk_points[:,1], -1),
            "z": np.expand_dims(bulk_points[:,2], -1),
         },
         outvar={
            "strain_norm": np.zeros((n_bulk_points,1)),
         },
         batch_size=n_bulk_points
      )
      return interior_condition



class PINNTrainer(ABC):
   def __init__(
         self,
         model,
         boundary_points,
         boundary_displacements,
         bulk_points,
         bulk_equations,
         n_bulk_samples=500, lr=1e-2, loss_stop=0.5
      ):
      self.eq = bulk_equations
      self.model = model
      self.nodes = self.eq.make_nodes() + [self.model.model.make_node(name=f"PINN_{self.eq.name}")]

      # Grab some random points from the interior
      indices = list(range(len(bulk_points)))
      shuffle(indices)
      self.bulk_points = bulk_points[indices[:n_bulk_samples]]
      self.n_bulk_points = n_bulk_samples

      # Store the initial boundary points
      self.boundary_points = boundary_points
      self.boundary_displacements = boundary_displacements
      self.n_boundary_points = len(self.boundary_points)

      self.optimizer = optim.Adam(model.model.parameters(), lr=lr)
      self.loss_stop = loss_stop
      self.loss_value = None
      self._create_domain()
      self._create_aggregator()
      self.domain.load_data()

   def _create_domain(self):
      self.domain = Domain()
      interior_condition = self.eq.create_interior_condition(self.nodes, self.bulk_points)
      boundary_condition = self._create_boundary_condition()
      self.domain.add_constraint(boundary_condition, "boundary condition")
      if interior_condition:
         self.domain.add_constraint(interior_condition, "interior PDE constraint")

   def _create_boundary_condition(self):
      # Specify boundary values and displacements
      boundary_condition = PointwiseConstraint.from_numpy(
         nodes=self.nodes,
         invar={
            "x": np.expand_dims(self.boundary_points[:,0], -1),
            "y": np.expand_dims(self.boundary_points[:,1], -1),
            "z": np.expand_dims(self.boundary_points[:,2], -1),
         },
         outvar={
            "u": np.expand_dims(self.boundary_displacements[:,0], -1),
            "v": np.expand_dims(self.boundary_displacements[:,1], -1),
            "w": np.expand_dims(self.boundary_displacements[:,2], -1),
         },
         batch_size=self.n_boundary_points
      )
      return boundary_condition

   def _create_aggregator(self):
      global_optimizer_model = self.domain.create_global_optimizer_model()
      self.agg = physicsnemo.sym.loss.aggregator.GradNorm(
         global_optimizer_model.parameters(),
         self.domain.get_num_losses()
      )

   def training_step(self, iteration):
      self.optimizer.zero_grad()
      losses = self.domain.compute_losses(iteration)
      agg_loss = self.agg.forward(losses, iteration)
      self.loss_value = agg_loss
      agg_loss.backward()
      self.optimizer.step()
      return agg_loss

   def converged(self):
      if self.loss_value.item() < self.loss_stop:
         return True
      return False
