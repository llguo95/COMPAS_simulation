import logging
import time
import f3dasm
import hydra
from compas_script import compas_function


@hydra.main(config_path=".", config_name="config")
def main(config):
    # If it is the first job in the array,
    # first create the designspace, then execute my_function on the designs.
    if int(config.slurm.arrayid) == 0:
        """Block 1: Design of Experiment"""

        # Filling the design space
        sampler = f3dasm.sampling.Sampler.from_yaml(config)
        data = f3dasm.ExperimentData.from_sampling(sampler)

        """Block 2: Data Generation"""

        # Execute the data generation function
        data.run(
            compas_function, mode='cluster',
            kwargs={
                "slurm_jobid": config.slurm_jobid,
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
            compas_function, mode='cluster',
            kwargs={
                "slurm_jobid": config.slurm_jobid,
            }
        )

    # Store the data generation function
    data.store()


if __name__ == "__main__":
    main()
