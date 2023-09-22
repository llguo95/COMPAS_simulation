import logging
import numpy as np
import os
import f3dasm
import shutil
import socket
import pandas as pd
import fileinput


def compas_function(design: f3dasm.Design, slurm_jobid=0):
    ddx = design.get('ddx')
    ddy = design.get('ddy')
    rrotz = design.get('rrotz')

    output = compas_objective(
        ddx, ddy, rrotz, jobnumber=design.job_number, slurm_jobid=slurm_jobid)
    design.set('acc_nlcr', output)
    return design


def compas_objective(ddx, ddy, rrotz, jobnumber=0, slurm_jobid=0):
    # Resource and work directories
    resources_directory = "/home/leoguo/GitHub/COMPAS_simulation/package_loc/COMPAS10"
    work_directory = resources_directory

    # Input file
    ansys_input_path_source = work_directory + "/subinput/submodel_run.txt"
    ansys_input_path = work_directory + \
        "/subinput/submodel_run_%s_%d.txt" % (str(slurm_jobid), jobnumber)
    shutil.copyfile(ansys_input_path_source, ansys_input_path)
    input_file = fileinput.FileInput(files=ansys_input_path, inplace=True)
    for line in input_file:
        if "ddx=" in line:  # ddx
            print('ddx=' + str(ddx), end='\n')
        elif "ddy=" in line:  # ddy
            print('ddy=' + str(ddy), end='\n')
        elif "rrotz=" in line:  # rrotz
            print('rrotz=' + str(rrotz), end='\n')
        else:
            print(line, end='')

    # Output file
    suboutput = 'suboutput_sample' if socket.gethostname() == 'hp' else 'suboutput'
    suboutput_directory = work_directory + '/' + suboutput
    if not os.path.isdir(suboutput_directory):
        os.mkdir(suboutput_directory)
    ansys_output_path = suboutput_directory + "/submodell_test.lis"

    # The command line
    cmdl = "ansys232 -b -dis -mpi openmpi -np 24 -g -i %s >& ansys_solve.out" % ansys_input_path

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

    # Cleaning up the output directory
    if socket.gethostname() != 'hp' and os.path.isdir(suboutput_directory):
        shutil.rmtree(suboutput_directory)

    return output
