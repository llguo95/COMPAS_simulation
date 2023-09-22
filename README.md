*Author: Leo Guo\
Latest update: 22 September 2023*

# Instructions
0. Clone this repository:
    ```
    git clone https://github.com/llguo95/COMPAS_simulation.git
    ```
1. Upload `COMPAS10` into `package_loc` (at the same level as `doe`).
    - Make sure the `COMPAS10/suboutput` folder is fully empty.
2. Set up a Python (3.7+) environment that contains all the packages indicated in `requirements.txt`.
    ```
    pip install -r requirements.txt
    ```
3. Enter into `package_loc/doe` and submit `slurmjob.sh`.