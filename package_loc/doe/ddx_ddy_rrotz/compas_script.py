import logging
import numpy as np
import os
import f3dasm
import shutil
import socket
import pandas as pd
import fileinput
from pathlib import Path


def compas_function(design: f3dasm.Design, slurm_jobid=0):
    ddx = design.get('ddx')
    ddy = design.get('ddy')
    rrotz = design.get('rrotz')

    logging.info("job id %s, job number %s, design %s" % (
        str(slurm_jobid), str(design.job_number), str([ddx, ddy, rrotz])))

    output = compas_objective(
        ddx, ddy, rrotz, jobnumber=design.job_number, slurm_jobid=slurm_jobid)
    design.set('acc_nlcr', output)
    return design


def compas_objective(ddx, ddy, rrotz, jobnumber=0, slurm_jobid=0, iteration_number=0,):
    # Resource and work directories
    resources_directory = str(
        Path(__file__).parent.parent.parent / "COMPAS10" / "subinput")
    work_directory = str(Path(__file__).parent.parent.parent / "COMPAS10" /
                         "suboutput" / "outputs" / str(slurm_jobid) / str(jobnumber) / str(iteration_number))

    # Input file
    ansys_input_path_source = resources_directory + "/submodel_run.txt"
    ansys_input_path = os.getcwd() + "/submodel_run_%s_%d.txt" % (str(slurm_jobid), jobnumber)
    shutil.copyfile(ansys_input_path_source, ansys_input_path)
    input_file = fileinput.FileInput(files=ansys_input_path, inplace=True)
    for line in input_file:
        if "ddx=" in line:  # ddx
            print('ddx=' + str(ddx), end='\n')
        elif "ddy=" in line:  # ddy
            print('ddy=' + str(ddy), end='\n')
        elif "rrotz=" in line:  # rrotz
            print('rrotz=' + str(rrotz), end='\n')
        elif "../subinput" in line:
            line = line.replace("../subinput", resources_directory)
            line = line.replace(",'.'", '')
            print(line, end='')
        else:
            print(line, end='')

    # Output file
    if not os.path.isdir(work_directory):
        os.makedirs(work_directory)

    ansys_output_path = work_directory + "/submodell_test.lis"

    # The command line
    cmdl = "ansys232 -dir %s -b -dis -mpi openmpi -np 6 -g -i %s >& %s/ansys_solve.out" % (
        work_directory, ansys_input_path, work_directory)

    # Running the command line
    if socket.gethostname() == 'hp':
        pass
    else:
        os.system(cmdl)

    # Reading the output
    if os.path.exists(ansys_output_path):
        output = pd.read_csv(ansys_output_path, sep='\s+').acc_nlcr.iloc[-1]
    else:
        logging.error("No output found.")
        output = np.nan

    # # Cleaning up the output directory
    # if socket.gethostname() != 'hp' and os.path.isdir(work_directory):
    #     shutil.rmtree(work_directory)

    return output
