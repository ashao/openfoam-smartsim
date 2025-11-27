#!/usr/bin/env python

import argparse
import os
import sys
import time

from pathlib import Path
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

from smartsim import Experiment
from smartsim.status import TERMINAL_STATUSES


platform_config = {
    "local": {
        "launcher": "local",
        "interface": "lo",
        "run_command": "mpirun"
    },
    "hotlum": {
        "launcher": "slurm",
        "interface": "bond0",
        "run_command": "srun"
    },
    "vader": {
        "launcher": "slurm",
        "interface": "bond0",
        "run_command": "srun"
    },
}

def main(args):

    # ----------------------------------------------------------------
    # Create the SmartSim experiment
    # ----------------------------------------------------------------

    input_case_path = Path(args.case)
    case_name = input_case_path.stem
    experiment_name = f"{args.experiment}_{case_name}_{args.mesh_solver_type}"
    if args.mesh_solver_type == "PINN":
        experiment_name = f"{experiment_name}_{args.pinn_type}"

    exp = Experiment(experiment_name, launcher=platform_config[args.platform]["launcher"])

    # ----------------------------------------------------------------
    # Launch the database
    # ----------------------------------------------------------------

    db = exp.create_database(port=8000, interface=platform_config[args.platform]["interface"])
    exp.generate(db, overwrite=True)

    # ----------------------------------------------------------------
    # Get the number of MPI ranks from system/decomposeParDict
    # ----------------------------------------------------------------

    # build the full path to the decomposeParDict
    decompose_dict_path = input_case_path / 'system' / 'decomposeParDict'

    # load the dictionary
    decompose = ParsedParameterFile(decompose_dict_path)

    # extract the numberOfSubdomains entry
    num_mpi_ranks = decompose['numberOfSubdomains']

    # ----------------------------------------------------------------
    # Configure and create the OpenFOAM mesh-motion model
    # ----------------------------------------------------------------

    # Create OpenFOAM moveDynamicMesh run settings
    openfoam_rs = exp.create_run_settings(
        exe="moveDynamicMesh",
        exe_args="-parallel",
        run_command=platform_config[args.platform]["run_command"]
    )
    openfoam_rs.set_tasks(num_mpi_ranks)
    openfoam_rs.set_nodes(1)

    # Create the model from the OpenFOAM case argument
    openfoam_model = exp.create_model(
        name=args.case,
        run_settings=openfoam_rs
    )
    openfoam_model.attach_generator_files(to_copy=str(input_case_path.absolute()))

    # ----------------------------------------------------------------
    # Configure and create the ML training model
    # ----------------------------------------------------------------

    training_rs = exp.create_run_settings(
        exe="python",
        exe_args=f"ml_model_training.py {num_mpi_ranks} {args.pinn_type}"
    )
    training_rs.set_tasks(1)
    training_rs.set_nodes(1)
    training_rs.set_cpus_per_task(128)

    ml_model_training = exp.create_model(
        name="ml_model_training",
        run_settings=training_rs
    )
    ml_model_training.attach_generator_files(
        to_copy=["ml_model_training.py", "networks/MLP.py", "networks/PINN.py"]
    )

    exp.generate(ml_model_training, overwrite=True)

    # ----------------------------------------------------------------
    # Run the experiment
    # ----------------------------------------------------------------

    try:
        exp.start(db)
        print(f"Database started at: {db.get_address()}")
        print("Running the OpenFOAM case")
        exp.generate(openfoam_model, overwrite=True)
        exp.start(openfoam_model, ml_model_training, block=False)

        while True:
            time.sleep(1)
            if exp.get_status(openfoam_model)[0] in TERMINAL_STATUSES:
                exp.stop(ml_model_training)
                break

    except Exception as e:
        print("Caught an exception:", e)

    finally:
        exp.stop(db)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a SmartSim Machine-Learning mesh deformation experiment"
    )
    parser.add_argument(
        "--experiment", "-e",
        default="meshMotion",
        help="Name of the SmartSim experiment (e.g., mesh_deformation)"
    )
    parser.add_argument(
        "--mesh-solver-type",
        default="PINN",
        choices=["Laplace", "PINN"],
        help="The solver type for mesh motion"
    )
    parser.add_argument(
        "--case", "-c",
        required=True,
        help="Name of the OpenFOAM case folder (e.g., ellipsoid3D)"
    )
    parser.add_argument(
        "--platform",
        default="local",
        help="The platform on which this is being run"
    )
    parser.add_argument(
        "--pinn-type",
        default=None,
        help="The type of PINN to use"
    )
    args = parser.parse_args()
    main(args)
