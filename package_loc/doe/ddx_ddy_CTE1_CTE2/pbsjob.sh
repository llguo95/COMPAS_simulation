 #!/bin/bash
 # Torque directives (#PBS) must always be at the start of a job script!
 #PBS -N compas_mfb
 #PBS -q guest
 #PBS -l nodes=1:ppn=4,walltime=12:00:00
 #PBS -t 0

 # Make sure I'm the only one that can read my output
 umask 0077

 JOB_ID=$(echo "${PBS_JOBID}" | sed 's/\[[^][]*\]//g')

 module load use.own
 module load anaconda3
 module load ansys/2023r1
 cd $PBS_O_WORKDIR

 # Here is where the application is started on the node
 # activating my conda environment:

 conda activate compas_env

 # limiting number of threads
 OMP_NUM_THREADS=12
 export OMP_NUM_THREADS=12


 # Check if PBS_ARRAYID exists, else set to 1
 if ! [ -n "${PBS_ARRAYID+1}" ]; then
   PBS_ARRAYID=None
 fi

 # Executing my python program

 python main_cluster.py ++hpc.jobid=${PBS_ARRAYID} hydra.run.dir=outputs/${now:%Y-%m-%d}/${JOB_ID}