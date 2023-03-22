import os
import time

def main():
    cmdl = "ansys2021r2 -b -dis -mpi openmpi -np ${SLURM_NTASKS} -g -i ./../Polymer_Optimization_Material_v02.dat >& ansys_solve.out"
    
    for i in range(3):
        start = time.time()
        os.system(command=cmdl)
        end = time.time()
        print()
        print('Iteration %d took %f seconds.' % (i, end - start))

if __name__ == "__main__":
    main()