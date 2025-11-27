import numpy as np
import torch
import torch.optim as optim

from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from smartredis import Client

from MLP import MLP, MLPTrainer
import PINN

import argparse
import io
import time
from pathlib import Path

bulk_points_key = lambda i: f"points_MPI_{i}"
distances_key = lambda i: f"distances_MPI_{i}"
displacements_key = lambda i: f"displacements_MPI_{i}"

default_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class MultiCriteriaEarlyStopping:
    def __init__(self, patience=40, relative_tolerance=0.01):
        """
        Tracks multiple criteria for early stopping based on relative improvement.

        Args:
            patience (int): Number of steps to wait without improvement before stopping per criterion.
            relative_tolerance (float): Minimum relative improvement required (e.g., 0.01 for 1%).
        """
        self.patience = patience
        self.relative_tolerance = relative_tolerance
        self.best_losses = {}      # Track best loss per component
        self.counters = {}         # Track patience counter per component

    def step(self, loss_dict):
        """
        Update early stopping state based on multiple validation losses.

        All losses are evaluated based on relative improvement:
        relative_improvement = (best_loss - current_loss) / best_loss

        Args:
            loss_dict (dict): e.g., {'boundary': 0.2, 'bulk': 0.5, 'pde': 1e-4}

        Returns:
            bool: True if ALL criteria have stagnated and exceeded patience threshold.
        """
        all_stagnant = True

        for key, loss in loss_dict.items():
            # Convert tensor to float if needed
            if hasattr(loss, "item"):
                loss = loss.item()

            # Initialize tracking for this component
            if key not in self.best_losses:
                self.best_losses[key] = loss
                self.counters[key] = 0
                all_stagnant = False
                continue

            # Compute relative improvement
            relative_improvement = (self.best_losses[key] - loss) / self.best_losses[key]

            # Check if improvement meets tolerance
            if relative_improvement >= self.relative_tolerance:
                # Improvement found, update best loss and reset counter
                self.best_losses[key] = loss
                self.counters[key] = 0
                all_stagnant = False
            else:
                # No improvement, increment patience counter
                self.counters[key] += 1
                if self.counters[key] < self.patience:
                    all_stagnant = False

        self.all_stagnant = all_stagnant

    def reset(self):
        """Reset the early stopping state."""
        self.best_losses.clear()
        self.counters.clear()

    def should_stop(self):
        """Check if all criteria have stagnated and exceeded patience."""
        return self.all_stagnant


class PatienceMonitor:
    def __init__(self, relative_tolerance=0.01, max_patience_steps=20):
        self.relative_tolerance = relative_tolerance
        self.max_patience_steps = max_patience_steps
        self.best_loss = np.inf
        self.counter = 0

    def step(self, loss):
        improvement = (self.best_loss - loss) / self.best_loss
        if improvement > self.relative_tolerance:
            self.counter = 0
        else:
            self.counter += 1
        if loss < self.best_loss:
            self.best_loss = loss

    def should_stop(self):
        return self.counter >= self.max_patience_steps

def retrieve_model(args, dimension):
    # Initialize the model
    if args.model_name == "mlp":
        model = MLP(
            input_size=dimension,
            output_size=dimension,
            num_layers=3,
            layer_width=10,
            activation_fn=torch.nn.ELU()
        )
    else:
        model = PINN.MLP(
            layer_size=20,
            nr_layers=2,
        )

    return model

def retrieve_trainer(args, model, X_boundary, X_bulk, n_bulk_samples, **kwargs):
    # Initalize the trainer from scratch each time
    try:
        eq = getattr(PINN, args.model_name)()
    except AttributeError:
        raise ValueError(f"Invalid bulk constraint: {args.model_name}")

    trainer = PINN.PINNTrainer(
        model, X_boundary, X_bulk, eq, n_bulk_samples=n_bulk_samples, **kwargs
    )
    return trainer

def retrieve_point_fields(client, mpi_ranks, key_constructor):
    point_field_by_rank = {r: client.get_tensor(key_constructor(r)) for r in mpi_ranks}
    point_field = np.vstack(list(point_field_by_rank.values()))
    start = 0
    indices = {}
    for r, rank_points in point_field_by_rank.items():
        end = start + rank_points.shape[0]
        indices[r] = np.arange(start, end)
        start = end

    return point_field, indices

def bc_stage(model, trainer, n_epochs):
    start = time.perf_counter()
    best_loss = np.inf
    patience_monitor = PatienceMonitor(relative_tolerance=0.01)
    for epoch in range(n_epochs):
        loss = trainer.step_bc_only()
        patience_monitor.step(loss)
        if loss < best_loss:
            best_state = model.state_dict()
            best_loss = loss
        if patience_monitor.should_stop():
            break
    train_time = time.perf_counter() - start
    print(
        f"Boundary Conditions: Loss {best_loss:.3e}, number of epochs {epoch}, time elapsed: {train_time:.3f}s",
        flush=True,
    )
    return best_state, epoch

def bulk_and_bc_stage(model, trainer, n_epochs):
    best_loss = np.inf
    patience_monitor = MultiCriteriaEarlyStopping(relative_tolerance=0.01)
    start = time.perf_counter()
    for epoch in range(n_epochs):
        _, _, validation_losses = trainer.step(epoch)
        sum_validation_losses = sum(validation_losses.values())
        patience_monitor.step(validation_losses)
        # Display progress
        if (epoch-1) % 10 == 0:
            print(
                f"\tEpoch {epoch-1} Aggregated Loss: {sum_validation_losses:.3e}",
                flush=True,
            )
            for k,v in validation_losses.items():
                print(f"\t\t{k}: {v:.3e}", flush=True)
        # Always store the best model
        if sum_validation_losses < best_loss:
            best_loss = sum_validation_losses
            best_state = model.state_dict()
        # Stop early either because target tolerance reached or patience has run out
        if patience_monitor.should_stop():
            break
    train_time = time.perf_counter() - start
    print(f"\tEpoch {epoch-1} Aggregated Loss: {best_loss:.3e}", flush=True)
    for k,v in validation_losses.items():
        print(f"\t\t{k}: {v:.3e}", flush=True)
    print(
        f"BC and Bulk: Loss {best_loss:.3e}, number of epochs {epoch}, time elapsed: {train_time:.3f}s",
        flush=True,
    )
    return best_state, epoch

def train(args):

    mpi_ranks = range(args.mpi_ranks)
    client = Client()

    # Pause until the OpenFOAM simulation has posted the boundary points
    points_ready = client.poll_key("points", 1, 10000)
    if not points_ready:
        raise Exception("'points' key not found. Simulation may have failed")
    dimension = int(client.get_tensor("solution_dim")[0])

    # Retrieve the boundary and bulk points
    points = client.get_tensor("points")
    bulk_points, rank_indices = retrieve_point_fields(client, mpi_ranks, bulk_points_key)
    distance_to_boundary, _ = retrieve_point_fields(client, mpi_ranks, distances_key)

    print(f"Solution dimension = {dimension} Number of Points={len(points)}", flush=True)

    # Scale all the inputs (if needed)
    boundary_points_scaled = points
    bulk_points_scaled = bulk_points

    # Convert all the interior points to a tensor for final inference
    bulk_points_for_inference = torch.from_numpy(bulk_points_scaled).float().to(args.device)

    model = retrieve_model(args, dimension)
    trainer = retrieve_trainer(
        args,
        model,
        boundary_points_scaled,
        bulk_points_scaled,
        n_bulk_samples=2000,
        loss_stop=1e-2,
    )

    timestep = 1
    while True:
        print("\n"+"-"*10)
        print(f"TIMESTEP {timestep}")

        # Block until the data is ready
        data_ready = client.poll_key("data_ready", 1, 10000)
        if (not data_ready):
            raise RuntimeError("Data not found in SmartRedis; aborting training.")

        displacements = client.get_tensor("displacements")
        client.delete_tensor("data_ready")

        trainer.set_boundary_displacements(displacements)
        mag = np.sum(displacements**2, axis=1)
        mag_avg = np.mean(mag[mag>0])
        print(f"Average magnitude of displacements: {mag_avg}")

        # Begin curriculum training
        # Stage: Boundary conditions only
        best_state, boundary_epochs = bc_stage(model, trainer, args.max_epochs)
        model.load_state_dict(best_state)
        # Stage: PDE bulk conditions and boundary conditions
        best_state, bc_bulk_epochs = bulk_and_bc_stage(model, trainer, args.max_epochs)
        model.load_state_dict(best_state)

        print(f"Training completed in {bc_bulk_epochs+boundary_epochs} epochs")

        start = time.perf_counter()
        # Put the model in evaluation mode and perform the inference for bulk points
        model.eval()
        model.load_state_dict(best_state)
        bulk_displacements = (
            model._impl.forward(bulk_points_for_inference)
            .detach()
            .to("cpu")
            .numpy()
            .astype(np.float64)
        )
        model.train()

        # Put all the displacements back into the database by rank
        for r in mpi_ranks:
            displacements_rank = bulk_displacements[rank_indices[r],...]
            client.put_tensor(displacements_key(r), displacements_rank)

        client.put_tensor("displacements_ready", np.array([0]))
        send_time = time.perf_counter() - start
        print(f"Solution sent in {send_time}s")
        # Increase CFD+ML iteration
        timestep += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for mesh motion")
    parser.add_argument("mpi_ranks", help="number of mpi ranks", type=int)
    parser.add_argument("model_name",
                        help="which model to use to calculate interior displacements",
                        type=str
    )
    parser.add_argument(
        "--max_epochs",
        help="Maximum number of training epochs per timestep",
        type=int,
        default=1000
        )
    parser.add_argument(
        "--device",
        help="The device to deploy the ML tasks on",
        default=default_device
    )
    args = parser.parse_args()

    train(args)
