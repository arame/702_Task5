#! /bin/bash

#SBATCH --job-name="Geoff P 702"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=geoffrey.payne@city.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output job%J.output
#SBATCH --error jo%J.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --partition=arti


module load cuda/10.0

python3 main.py
