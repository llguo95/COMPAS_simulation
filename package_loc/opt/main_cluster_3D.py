import itertools
import logging
import time
import f3dasm
import hydra
import numpy as np
from compas_optimization_3D import compas_opt_function
import pandas as pd
# import itertools
# import numpy as np


@hydra.main(config_path=".", config_name="config_3D")
def main(config):
    logging.info("Main method started.")
    # If it is the first job in the array,
    # first create the designspace, then execute my_function on the designs.
    if int(config.slurm.arrayid) == 0:
        """Block 1: Design of Experiment"""

        # Filling the design space
        # sampler = f3dasm.sampling.Sampler.from_yaml(config)
        # data = f3dasm.ExperimentData.from_sampling(sampler)

        design = f3dasm.Domain.from_yaml(yaml=config.design)

        data = f3dasm.ExperimentData(design=design)

        # df_input = pd.read_csv(
        #     "/home/leoguo/GitHub/MFB/studies/spinodal/opt/custom_doe_opt.csv",
        #     header=[0, 1], index_col=0,
        # )

        df_input = pd.DataFrame(
            columns=data.data.data.columns[:3],
            data=list(itertools.product(
                *[parameter.categories for _, parameter in design.input_space.items()])),
        )

        df_input[('output', 'ddx_rec')] = np.nan
        df_input[('output', 'ddy_rec')] = np.nan
        df_input[('output', 'CTE_2_rec')] = np.nan
        df_input[('output', 'acc_nlcr_rec')] = np.nan

        data.add(data=df_input.reset_index(drop=True))

        # Save input data
        input_data = data.get_input_data()
        input_data.to_csv('input.csv')

        """Block 2: Data Generation"""

        # Execute the data generation function
        data.run(
            compas_opt_function, mode='cluster',
            kwargs={
                "slurm_jobid": config.slurm.jobid,
                "hyperparameters": config.hyperparameters,
            }
        )

    # In any other case, the design has already been made
    # Therefore, load it from disk and run my_function on it.
    elif int(config.slurm.arrayid) > 0:
        # Retrieve the file from disk
        data = None
        while data is None:
            try:
                data = f3dasm.ExperimentData.from_file()
            except:
                logging.error("Data not found. Retrying in 5 seconds...")
                time.sleep(5)
                pass

        data.run(
            compas_opt_function, mode='cluster',
            kwargs={
                "slurm_jobid": config.slurm.jobid,
                "hyperparameters": config.hyperparameters,
            }
        )

    # Store the data generation function
    # data.store(filename='spinodal_doe')


if __name__ == "__main__":
    main()
