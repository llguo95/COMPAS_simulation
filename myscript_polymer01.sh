#!/bin/bash -l

#SBATCH -D ./ansysPolymer01/       				# directory where the mpiexec command is ran
#SBATCH -J "ansysPolymer01"            			# name of the job (can be change to whichever name you like)
#SBATCH --get-user-env             			# to set environment variables

#SBATCH -o DHPC_output_%j.txt         			# output file (DHPC environment)
#SBATCH -e DHPC_error_%j.txt          			# error file (DHPC environment)

#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --account=innovation

myjobid=$SLURM_JOB_ID
echo "myjobid = " $myjobid

# Load modules:
module load 2022r2
module load openmpi
module load ansys/2021R2
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate COMPAS_env
srun python3 ./../run_ansys_test.py
conda deactivate

tail -20 ansys_solve.out