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

class PatienceMonitor:
    def __init__(self, relative_tolerance=0.01, max_patience_steps=50):
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

def retrieve_trainer(args, model, X_int, X, y, **kwargs):
    # Initalize the trainer from scratch each time
    if args.model_name == "mlp":
        trainer = MLPTrainer(model, X, y, radius_power=args.radius_power)
        return trainer
    try:
        eq = getattr(PINN, args.model_name)()
    except AttributeError:
        raise ValueError(f"Invalid bulk constraint: {args.model_name}")

    trainer = PINN.PINNTrainer(model, X, y, X_int, eq, **kwargs)
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
    interior_points, rank_indices = retrieve_bulk_points(client, mpi_ranks)

    X_norm = np.max(np.abs(points))
    X_norm = 1.
    print(f"Solution dimension = {dimension}.", flush=True)
    print(f"X_norm = {X_norm}.", flush=True)

    # Convert to tensors
    X = torch.from_numpy(points).to(torch.float)/X_norm
    X_bulk = torch.from_numpy(interior_points).to(torch.float)/X_norm
    X_bulk_gpu= torch.from_numpy(interior_points).to(torch.float).to("cuda")/X_norm

    state_dict = None
    model = retrieve_model(args, dimension)

    iteration = 1
    while True:

        print (f"Iteration {iteration}")

        # Block until the data is ready
        data_ready = client.poll_key("data_ready", 1, 10000)
        if (not data_ready):
            raise RuntimeError("Data not found in SmartRedis; aborting training.")

        displacements = client.get_tensor("displacements")
        client.delete_tensor("data_ready")

        y = torch.from_numpy(displacements).to(torch.float)

        trainer = retrieve_trainer(args, model, X_bulk, X, y, n_bulk_samples=5000)
        patience_monitor = PatienceMonitor()
        best_loss = np.inf

        start = time.perf_counter()

        for epoch in range(args.max_epochs):
            loss = trainer.training_step(epoch)
            patience_monitor.step(loss)
            # Display progress
            if (epoch-1) % 10 == 0:
                print(loss.item(), flush=True)
            # Always store the best model
            if loss < best_loss:
                best_loss = loss
                best_state = model.model.state_dict()
            # Stop early either because target tolerance reached or patience has run out
            if trainer.converged() or patience_monitor.should_stop():
                break

        train_time = time.perf_counter() - start
        print(f"Loss {best_loss}, number of epochs {epoch}, time elapsed: {train_time:.3f}s", flush=True)

        # Put the model in evaluation mode and perform the inference for bulk points
        model.model.eval()
        model.model.load_state_dict(best_state)
        bulk_displacements = model(X_bulk_gpu).detach().to("cpu").numpy().astype(np.float64)
        model.model.train()
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
    args = parser.parse_args()

    train(args)
