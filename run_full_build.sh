#!/bin/bash
#SBATCH --job-name=instruct-retro-build     # create a short name for your job
#SBATCH --nodes=1                # total number of nodes
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --output=./logs/%x_%j.log  # control where the stdout will be
#SBATCH --error=./logs/%x_%j.err   # control where the error messages will be
#SBATCH --reservation=todi


# Run main script.
srun -ul --reservation=todi --environment=instruct-retro-faiss-cpu bash -c "
  # Change cwd and run the main training script.
  cd $SCRATCH/workspace/Megatron-LM
  # run complete build
  bash tools/retro/examples/20240730_preprocess_data.sh build
"
