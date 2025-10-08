
import numpy as np
import physicsnemo.sym
import physicsnemo.sym.loss.aggregator
import torch
import torch.optim as optim

from sympy import Symbol, Function
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseConstraint, PointwiseInteriorConstraint
from physicsnemo.sym.hydra import instantiate_arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver

# Override unnecessary cuda dependencies
import torch.cuda.nvtx
torch.cuda.nvtx.range_push = lambda *_, **__: None
torch.cuda.nvtx.range_pop = lambda *_, **__: None

class NavierLame3d(PDE):
    def __init__(self, lam, mu):
        # Position variables
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # Displacements as a function of x
        u = Function("u")(x, y, z)
        v = Function("v")(x, y, z)
        w = Function("w")(x, y, z)

        # Define terms of PDEs
        div_u = u.diff(x) + v.diff(y) + w.diff(z)

        laplace_u = u.diff(x,2) + u.diff(y,2) + u.diff(z,2)
        laplace_v = v.diff(x,2) + v.diff(y,2) + v.diff(z,2)
        laplace_w = w.diff(x,2) + w.diff(y,2) + w.diff(z,2)

        self.equations = {
            "mom_x": mu*laplace_u + (lam + mu)*div_u.diff(x),
            "mom_y": mu*laplace_v + (lam + mu)*div_u.diff(y),
            "mom_z": mu*laplace_w + (lam + mu)*div_u.diff(z),
        }

class ElasticModel:
   def __init__(self, layer_size=20, nr_layers=3):
      self.model = FullyConnectedArch(
          input_keys=[Key("x"), Key("y"), Key("z")],
          output_keys=[Key("u"), Key("v"), Key("w")],
          layer_size=20,
          nr_layers=3
      )

class ElasticTrainer:
   def __init__(self, model, interior_points, boundary_points, boundary_displacements, n_interior_samples=500, mu=0.1, lam=0.3, lr=1e-3, loss_stop=1e-2):
      self.nl_3d_eq = NavierLame3d(mu, lam)
      self.model = model
      self.nodes = self.nl_3d_eq.make_nodes() + [self.model.model.make_node(name="nl_3d_net")]

      # Grab some random points from the interior
      self.interior_points = interior_points
      self.n_interior_points = len(self.interior_points)

      # Store the initial boundary points
      self.boundary_points = boundary_points
      self.boundary_displacements = boundary_displacements
      self.n_boundary_points = len(self.boundary_points)

      self.optimizer = optim.Adam(model.model.parameters(), lr=lr)
      self.loss_stop = loss_stop
      self.loss_value = None
      self._create_domain()

   def _create_domain(self):
      self.domain = Domain()

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

      # Enforce the Navier-Lame equations in the interior
      interior_condition = PointwiseConstraint.from_numpy(
         nodes=self.nodes,
         invar={
            "x": np.expand_dims(self.interior_points[:,0], -1),
            "y": np.expand_dims(self.interior_points[:,1], -1),
            "z": np.expand_dims(self.interior_points[:,2], -1),
         },
         outvar={
            "mom_x": np.zeros((self.n_interior_points,1)),
            "mom_y": np.zeros((self.n_interior_points,1)),
            "mom_z": np.zeros((self.n_interior_points,1)),
         },
         batch_size=self.n_interior_points
      )

      self.domain.add_constraint(boundary_condition, "boundary condition")
      self.domain.add_constraint(interior_condition, "interior PDE constraint")
      self.domain.load_data()

      global_optimizer_model = self.domain.create_global_optimizer_model()
      self.agg = physicsnemo.sym.loss.aggregator.Sum(
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
      return agg_loss, self.model.model._impl

   def converged(self):
      if self.loss_value.item() < self.loss_stop:
         return True
      return False




