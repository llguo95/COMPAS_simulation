*Author: Leo Guo\
Latest update: 24 September 2023*

# Instructions
0. Set up a Python (3.7+) environment.
    - If you do not have a package manager yet, I recommend [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Please follow the quick command line install instructions under the Linux tab. Beware to set the destination folder correctly, as the instructions will install miniconda in the home (`~/`) folder. By removing this prefix, you will install it inside the folder you currently have opened in your terminal.
    - After the installation procedure, open a new terminal. If the installation went correctly, you will see that the base conda environment has been activated at the beginning of each command line:
        ```
        (base) [<username>@<device> <directory>]$
        ```
    - Set up a new environment that is used for the data-driven pipeline:
        ```
        conda env create --name compas_env python=3.8
        ```
    - Activate the environment:
        ```
        conda activate compas_env
        ```
1. Clone this repository:
    ```
    git clone https://github.com/llguo95/COMPAS_simulation.git
    ```
2. Enter into the `COMPAS_simulation` directory and install all the required packages. To do this, run:
    ```
    pip install -r requirements.txt
    ```
3. Upload `COMPAS10` into `package_loc` (at the same level as `doe`).
    - Make sure the `COMPAS10/suboutput` folder is fully empty.
4. Enter into `package_loc/doe` and submit `slurmjob.sh`.
5. You will see outputs appear in the `outputs` directory.