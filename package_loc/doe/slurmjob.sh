#!/bin/bash -l

#SBATCH -D ./../COMPAS10/suboutput/ 					# directory where the mpiexec command is ran
#SBATCH -J "compas_mfb"            			# name of the job (can be change to whichever name you like)
#SBATCH --get-user-env             			# to set environment variables

#SBATCH -o DHPC_output_%j.txt         			# output file (DHPC environment)
#SBATCH -e DHPC_error_%j.txt          			# error file (DHPC environment)

#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --account=research-eemcs-me

myjobid=$SLURM_JOB_ID
echo "myjobid = " $myjobid

module load 2022r2
module load openmpi
module load ansys/2023R2
# ansys232 -b -dis -mpi openmpi -np ${SLURM_NTASKS} -g -i ./../subinput/submodel_run.txt >& ansys_solve.out
python3 main_cluster.py ++slurm.arrayid=${SLURM_ARRAY_TASK_ID} hydra.run.dir=outputs/${now:%Y-%m-%d}/${myjobid} ++slurm_jobid=${myjobid}

# tail -20 ansys_solve.out

