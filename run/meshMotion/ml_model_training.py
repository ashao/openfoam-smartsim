import numpy as np
import torch
import torch.optim as optim

from matplotlib import pyplot as plt
from smartredis import Client

from MLP import MLP, MLPTrainer
import PINN

import argparse
import io
from pathlib import Path

point_key = lambda i: f"points_MPI_{i}"
displacements_key = lambda i: f"displacements_MPI_{i}"

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
            layer_size=5,
            nr_layers=2,
        )

    return model

def retrieve_trainer(args, model, X_int, X, y):
    # Initalize the trainer from scratch each time
    if args.model_name == "mlp":
        trainer = MLPTrainer(model, X, y, radius_power=args.radius_power)
        return trainer
    try:
        eq = getattr(PINN, args.model_name)()
    except AttributeError:
        raise ValueError(f"Invalid bulk constraint: {args.model_name}")

    trainer = PINN.PINNTrainer(model, X, y, X_int, eq)
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

    # Read the solution direction from a database

    # Pause until the OpenFOAM simulation has posted the boundary points
    points_ready = client.poll_key("points", 1, 10000)
    if not points_ready:
        raise Exception("'points' key not found. Simulation may have failed")
    dimension = int(client.get_tensor("solution_dim")[0])
    print (f"Solution dimension = {dimension}.", flush=True)
    model = retrieve_model(args, dimension)

    # Retrieve the boundary and bulk points
    points = client.get_tensor("points")
    interior_points, rank_indices = retrieve_bulk_points(client, mpi_ranks)

    # Convert to tensors
    X = torch.from_numpy(points).to(torch.float)
    X_bulk = torch.from_numpy(interior_points).to(torch.float)

    epochs = 100
    iteration = 1
    while True:

        print (f"Iteration {iteration}")

        # Block until the data is ready
        data_ready = client.poll_key("data_ready", 1, 10000)
        if (not data_ready):
            raise RuntimeError("Data not found in SmartRedis; aborting training.")

        displacements = client.get_tensor("displacements")
        np.savez(f"data_{iteration:03d}.npz", points, interior_points, displacements)
        client.delete_tensor("data_ready")
        y = torch.from_numpy(displacements).to(torch.float)

        trainer = retrieve_trainer(args, model, X_bulk, X, y)

        for epoch in range(epochs):
            loss, model_trained = trainer.training_step(epoch)
            if (epoch-1) % 10 == 0:
                print(loss.item(), flush=True)
            if trainer.converged():
                break

        print(f"MSE {loss.item()}, number of epochs {epoch}", flush=True)
        np.savez(
            f"data_{iteration:02d}.npz",
            points=points,
            displacements=displacements,
        )

        # Put the model in evaluation mode and perform the inference for bulk points
        model_trained.eval()
        bulk_displacements = model(X_bulk).detach().to("cpu").numpy().astype(np.float64)
        # Put all the displacements back into the database by rank
        for r in mpi_ranks:
            displacements_rank = bulk_displacements[rank_indices[r],...]
            client.put_tensor(displacements_key(r), displacements_rank)

        client.put_tensor("displacements_ready", np.array([0]))

        # Increase CFD+ML iteration
        iteration = iteration + 1

        # Check final iteration index and break
        if client.poll_key("final_iteration", 10, 10):
           print ("final iteration reached.")
           break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for mesh motion")
    parser.add_argument("mpi_ranks", help="number of mpi ranks", type=int)
    parser.add_argument("radius_power", help="power law to weight losses", type=float)
    parser.add_argument("model_name",
                        help="which model to use to calculate interior displacements",
                        type=str
    )
    args = parser.parse_args()

    train(args)
