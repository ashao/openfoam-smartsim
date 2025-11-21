import numpy as np
import torch
import torch.optim as optim

from matplotlib import pyplot as plt
from smartredis import Client

from MLP import MLP, MLPTrainer
import PINN

import argparse
import io
import time
from pathlib import Path

point_key = lambda i: f"points_MPI_{i}"
displacements_key = lambda i: f"displacements_MPI_{i}"

default_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

class PatienceMonitor:
    def __init__(self, relative_tolerance=0.01, max_patience_steps=20):
        self.relative_tolerance=relative_tolerance
        self.max_patience_steps = max_patience_steps
        self.best_loss = np.inf
        self.counter = 0

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

    def should_stop(self):
        return self.counter == self.max_patience_steps

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
            layer_size=10,
            nr_layers=3,
        )

    return model

def retrieve_trainer(args, model, X_boundary, X_bulk, **kwargs):
    # Initalize the trainer from scratch each time
    try:
        eq = getattr(PINN, args.model_name)()
    except AttributeError:
        raise ValueError(f"Invalid bulk constraint: {args.model_name}")

    trainer = PINN.PINNTrainer(model, X_boundary, X_bulk, eq, **kwargs)
    return trainer

def retrieve_bulk_points(client, mpi_ranks):
    bulk_points_by_rank = {r: client.get_tensor(point_key(r)) for r in mpi_ranks}
    bulk_points = np.vstack(list(bulk_points_by_rank.values()))
    start = 0
    indices = {}
    for r, rank_points in bulk_points_by_rank.items():
        end = start + rank_points.shape[0]
        indices[r] = np.arange(start, end)
        start = end

    return bulk_points, indices


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
    bulk_points, rank_indices = retrieve_bulk_points(client, mpi_ranks)

    X_norm = np.max(np.abs(points))
    X_norm = 1.
    print(f"Solution dimension = {dimension}", flush=True)
    print(f"X_norm = {X_norm}", flush=True)

    # Scale all the inputs (if needed)
    boundary_points_scaled = points/X_norm
    bulk_points_scaled = bulk_points/X_norm

    # Convert all the interior points to a tensor for final inference
    bulk_points_for_inference = torch.from_numpy(bulk_points_scaled).float().to(args.device)

    model = retrieve_model(args, dimension)
    trainer = retrieve_trainer(
        args,
        model,
        boundary_points_scaled,
        bulk_points_scaled,
        n_bulk_samples=1000,
        loss_stop=1e-2,
    )

    iteration = 1
    while True:

        print (f"Iteration {iteration}")

        # Block until the data is ready
        data_ready = client.poll_key("data_ready", 1, 10000)
        if (not data_ready):
            raise RuntimeError("Data not found in SmartRedis; aborting training.")

        displacements = client.get_tensor("displacements")
        client.delete_tensor("data_ready")

        trainer.set_boundary_displacements(displacements)
        start = time.perf_counter()

        # Begin curriculum training
        # Curriculum 1: Just boundary conditions
        best_loss = np.inf
        patience_monitor = PatienceMonitor()
        for epoch1 in range(args.max_epochs):
            loss = trainer.step_bc_only()
            patience_monitor.step(loss)
            if loss < best_loss:
                best_state = model.state_dict()
                best_loss = loss
            if patience_monitor.should_stop():
                break
        model.load_state_dict(best_state)
        train_time = time.perf_counter() - start
        print(f"Curriculum 1: Loss {best_loss}, number of epochs {epoch1}, time elapsed: {train_time:.3f}s", flush=True)

        # Curriculum 2: PDE bulk conditions and boundary conditions
        best_loss = np.inf
        patience_monitor = PatienceMonitor()
        start = time.perf_counter()
        for epoch2 in range(args.max_epochs):
            agg_loss, losses = trainer.step(epoch2)
            patience_monitor.step(agg_loss)
            # Display progress
            if (epoch2-1) % 10 == 0:
                print(f"Epoch {epoch2-1} Aggregated Loss: {agg_loss.item():.3e}")
                for k,v in losses.items():
                    print(f"\t{k}: {v.item():.3e}")
            # Always store the best model
            if agg_loss < best_loss:
                best_loss = agg_loss
                best_state = model.state_dict()
            # Stop early either because target tolerance reached or patience has run out
            if trainer.converged() or patience_monitor.should_stop():
                break
        train_time = time.perf_counter() - start
        print(f"Curriculum 2: Loss {best_loss}, number of epochs {epoch2}, time elapsed: {train_time:.3f}s", flush=True)

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

        # Increase CFD+ML iteration
        iteration += 1

        # Check final iteration index and break
        if client.poll_key("final_iteration", 10, 10):
            print ("final iteration reached.")
            break

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
