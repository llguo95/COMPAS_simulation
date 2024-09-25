#!/bin/bash -l

#SBATCH -J "compas_saltelli_doe_3D"            			# name of the job (can be change to whichever name you like)
#SBATCH --get-user-env             			# to set environment variables

#SBATCH -o outputs/DHPC_output_%A_%a.txt         			# output file (DHPC environment)
#SBATCH -e outputs/DHPC_error_%A_%a.txt          			# error file (DHPC environment)

#SBATCH --partition=compute
#SBATCH --time=100:00:00
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --account=research-eemcs-me
#SBATCH --array=0-9

conda activate compas_env

myjobid=$SLURM_ARRAY_JOB_ID
echo "myjobid = " $myjobid

module load 2024r1
module load openmpi
module load ansys/2023R2

python3 main_cluster.py ++slurm.arrayid=${SLURM_ARRAY_TASK_ID} ++slurm.jobid=${myjobid} hydra.run.dir=outputs/${myjobid}