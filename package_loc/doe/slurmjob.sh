#!/bin/bash -l

#SBATCH -J "compas_mfb"            			# name of the job (can be change to whichever name you like)
#SBATCH --get-user-env             			# to set environment variables

#SBATCH -o DHPC_output_%A_%a.txt         			# output file (DHPC environment)
#SBATCH -e DHPC_error_%A_%a.txt          			# error file (DHPC environment)

#SBATCH --partition=compute
#SBATCH --time=100:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --account=research-eemcs-me
#SBATCH --array=0-2

conda activate compas_env

myjobid=$SLURM_ARRAY_JOB_ID
echo "myjobid = " $myjobid

module load 2022r2
module load openmpi
module load ansys/2023R2
# ansys232 -b -dis -mpi openmpi -np ${SLURM_NTASKS} -g -i ./../subinput/submodel_run.txt >& ansys_solve.out
python3 main_cluster.py ++slurm.arrayid=${SLURM_ARRAY_TASK_ID} ++slurm.jobid=${myjobid} hydra.run.dir=outputs/${myjobid}

# tail -20 ansys_solve.out

