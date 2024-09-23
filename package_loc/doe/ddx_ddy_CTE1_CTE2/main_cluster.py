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
    if config.hpc.jobid == 0 and int(config.slurm.arrayid) == 0:
        """Block 1: Design of Experiment"""

        # Building (Saltelli) sampler
        sampler = f3dasm.sampling.Sampler.from_yaml(config)

        # Sampling from the sampler
        data_gen = f3dasm.ExperimentData.from_sampling(sampler)

        # Set to True if you want to filter your doe
        filter = False

        if filter:
            design = f3dasm.Domain.from_yaml(yaml=config.design)
            data = f3dasm.ExperimentData(design=design)
            data.add_numpy_arrays(data_gen.data.data.iloc[:1].input)
        else:
            data = data_gen

        input_data = data.get_input_data()
        input_data.to_csv('input.csv')

        """Block 2: Data Generation"""

        # Execute the data generation function

        while True:
            try:
                data.run(
                    compas_function, mode='cluster',
                    kwargs={
                        # "job_id": config.slurm.jobid,
                        "job_id": config.pbs_jobid,
                    }
                )
            except:
                continue
            else:
                break

    # In any other case, the design has already been made
    # Therefore, load it from disk and run my_function on it.
    elif not (config.hpc.jobid == 0 and int(config.slurm.arrayid) == 0):
        time.sleep(int(config.hpc.jobid) + int(config.slurm.arrayid))
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

        while True:
            try:
                data.run(
                    compas_function, mode='cluster',
                    kwargs={
                        # "job_id": config.slurm.jobid,
                        "job_id": config.pbs_jobid,
                    }
                )
            except:
                continue
            else:
                break

    # # Store the data generation function
    # data.store()


if __name__ == "__main__":
    main()
