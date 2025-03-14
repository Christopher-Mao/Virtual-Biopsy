#!/bin/bash
#SBATCH --job-name=python_job       # Create a short name for your job, required
#SBATCH --nodes=1                   # Number of nodes, required
#SBATCH --ntasks=1                 # Number of CPU cores to use, required
#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00             # Time limit hrs:min:sec, required
#SBATCH --output jobrecs/myjob.o%j          # Name of the stdout output file, required
#SBATCH --error jobrecs/myjob.e%j           # Name of stderr error file 
#SBATCH --partition=gpu1v100        # Specify the name of the GPU partition, required
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate myenv  # Activate your conda environment from dir
python pix2pix.py

