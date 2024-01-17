import logging
import time
import f3dasm
import hydra
from compas_script import compas_function


@hydra.main(config_path=".", config_name="config")
def main(config):
    logging.info("Main method started.")

    # If it is the first job in the array,
    # first create the designspace, then execute my_function on the designs.
    if int(config.slurm.arrayid) == 0:
        """Block 1: Design of Experiment"""

        # Filling the design space
        sampler = f3dasm.sampling.Sampler.from_yaml(config)
        # Uncomment this if you want to use the full data set defined in the config
        data = f3dasm.ExperimentData.from_sampling(sampler)

        # samples = f3dasm.ExperimentData.from_sampling(sampler)

        # # Comment this block if you don't want to filter your doe
        # design = f3dasm.Domain.from_yaml(yaml=config.design)
        # data = f3dasm.ExperimentData(design=design)
        # data.add(samples.data.data.loc[30:100].reset_index(drop=True))
        # #

        """Block 2: Data Generation"""

        # Execute the data generation function
        data.run(
            compas_function, mode='cluster',
            kwargs={
                "slurm_jobid": config.slurm.jobid,
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
                logging.info("Data found. Executing data.run")
            except:
                logging.error("Data not found. Retrying in 5 seconds...")
                time.sleep(5)
                pass

        data.run(
            compas_function, mode='cluster',
            kwargs={
                "slurm_jobid": config.slurm.jobid,
            }
        )

    # # Store the data generation function
    # data.store()


if __name__ == "__main__":
    main()
