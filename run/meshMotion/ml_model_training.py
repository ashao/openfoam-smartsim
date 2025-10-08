import argparse
import torch
import numpy as np
import io
import torch.optim as optim

from matplotlib import pyplot as plt
from smartredis import Client

from MLP import MLP, MLPTrainer
from elasticPINN import ElasticModel, ElasticTrainer


def train(args):
    client = Client()

    # Read the solution direction from a database
    dimension = int(client.get_tensor("solution_dim")[0])

    print (f"Solution dimension = {dimension}.")
    # Initialize the model
    if args.model_name == "mlp":
        model = MLP(
            input_size=dimension,
            output_size=dimension,
            num_layers=3,
            layer_width=10,
            activation_fn=torch.nn.ELU()
        )
    elif args.model_name == "elastic":
        model = ElasticModel(
            layer_size=20,
            nr_layers=3
        )

    data_ready = client.poll_key("points", 1, 10000)
    points = client.get_tensor("points")
    interior_points = np.vstack([client.get_tensor(f"points_MPI_{i}") for i in range(4)])
    np.random.shuffle(interior_points)
    X = torch.from_numpy(points).to(torch.float)
    X_int = torch.from_numpy(interior_points).to(torch.float)
    # Make sure all datasets are avaialble in the smartredis database.

    epochs = 5000
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

        # Initalize the trainer from scratch each time
        if args.model_name == "mlp":
            trainer = MLPTrainer(model, X, y, radius_power=args.radius_power)
        elif args.model_name == "elastic":
            trainer = ElasticTrainer(model, X_int, X, y)

        for epoch in range(epochs):
            loss, model = trainer.training_step(epoch)
            if (epoch-1) % 50 == 0:
                print(loss.item())
            if trainer.converged():
                break

        print(f"MSE {loss.item()}, number of epochs {epoch}", flush=True)
        np.savez(
            f"data_{iteration:02d}.npz",
            points=points,
            displacements=displacements,
        )

        # Store the model into SmartRedis
        # Put the model in evaluation mode.
        model.eval() # TEST
        model.double()
        # Prepare a sample input
        example_forward_input = torch.rand(dimension)
        # Convert the PyTorch model to TorchScript
        model_script = torch.jit.trace(model, example_forward_input)
        # Save the TorchScript model to a buffer
        model_buffer = io.BytesIO()
        torch.jit.save(model_script, model_buffer)
        # Set the model in the SmartRedis database
        print("Saving model")
        client.set_model("model", model_buffer.getvalue(), "TORCH", "CPU")
        client.put_tensor("model_ready", np.array([0]))
        model.float()

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
                        choices=["mlp", "elastic"],
                        type=str
    )
    args = parser.parse_args()

    train(args)
