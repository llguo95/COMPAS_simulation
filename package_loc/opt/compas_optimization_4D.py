import logging
import socket
import mfb
import gpytorch
import f3dasm
import numpy as np
import pandas as pd
import os
import torch
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(Path(__file__).parent.parent / "doe" / "ddx_ddy_CTE1_CTE2"))  # NOQA
from compas_script import compas_objective  # NOQA


def compas_opt_function(design: f3dasm.Design, hyperparameters: dict, slurm_jobid):
    logging.info("Optimization function wrapper started.")
    data_initial_doe = pd.read_csv(
        # Path(__file__).parent / "initial_doe_data_res_mf.csv",
        # Path(__file__).parent / "doe_data_seed2.csv",
        Path(__file__).parent / "doe_data_4D.csv",
        header=[0, 1], index_col=0
    ).dropna()

    # design
    regression_type = design.get('regression_type')
    regression_covar_base_name = design.get('regression_covar_base_name')
    optimization_acquisition_type = design.get('optimization_acquisition_type')

    # hyper
    data_dimensionality = hyperparameters.data.dimensionality
    data_fidelity_parameter_name = hyperparameters.data.fidelity_parameter_name
    data_low_fidelity_parameter = hyperparameters.data.low_fidelity_parameter
    data_high_fidelity_parameter = hyperparameters.data.high_fidelity_parameter
    data_initial_doe_size_hf = hyperparameters.data.data_initial_doe_size_hf
    optimization_lf_cost = hyperparameters.optimization.lf_cost
    optimization_iterations = hyperparameters.optimization.iterations
    optimization_budget = hyperparameters.optimization.budget

    result = compas_opt(
        data_initial_doe=data_initial_doe,
        data_dimensionality=data_dimensionality,
        data_fidelity_parameter_name=data_fidelity_parameter_name,
        data_low_fidelity_parameter=data_low_fidelity_parameter,
        data_high_fidelity_parameter=data_high_fidelity_parameter,
        data_initial_doe_size_hf=data_initial_doe_size_hf,
        regression_type=regression_type,
        regression_covar_base_name=regression_covar_base_name,
        optimization_acquisition_type=optimization_acquisition_type,
        optimization_lf_cost=optimization_lf_cost,
        optimization_iterations=optimization_iterations,
        optimization_budget=optimization_budget,
        jobnumber=design.job_number,
        slurm_jobid=slurm_jobid,
    )

    x_rec = result['x_rec'].flatten()
    y_rec = result['y_rec']

    design.set('ddx_rec', x_rec[0])
    design.set('ddy_rec', x_rec[1])
    design.set('CTE_1_rec', x_rec[2])
    design.set('CTE_2_rec', x_rec[3])
    design.set('acc_nlcr_rec', y_rec)

    return design


def compas_opt(
        data_initial_doe,
        data_dimensionality,
        data_fidelity_parameter_name,
        data_low_fidelity_parameter,
        data_high_fidelity_parameter,
        data_initial_doe_size_hf,
        regression_type,
        regression_covar_base_name,
        optimization_acquisition_type,
        optimization_lf_cost,
        optimization_iterations,
        optimization_budget,
        jobnumber,
        slurm_jobid,
):
    logging.info("Optimization function started.")

    #######################
    # Parameters: level 1 #
    #######################

    # Regression-related
    # meta
    regression_training_iter = 150
    regression_opt_kwargs = dict(lr=0.1)
    regression_opt_class = torch.optim.Adam
    regression_likelihood_class = gpytorch.likelihoods.GaussianLikelihood
    regression_mean_base_class = gpytorch.means.ZeroMean

    # Optimization-related
    # meta
    optimization_maximize = False

    #######################
    # Parameters: level 2 #
    #######################

    # Regression-related
    # derivative
    regression_covar_base_class = getattr(
        gpytorch.kernels, regression_covar_base_name
    )

    #######################
    # Parameters: level 3 #
    #######################

    # Regression-related
    # derivative
    regression_parameters_class = getattr(
        mfb.machinelearning.gpr, regression_type + '_Parameters'
    )
    regressor_class = getattr(mfb.machinelearning.gpr, regression_type)

    if regression_type in ["Cokgj", "Cokgd"]:
        # Note: no fidelity mixture
        regression_mean = torch.nn.ModuleList([
            regression_mean_base_class(),
            regression_mean_base_class(),
        ])
        # Note: no fidelity mixture
        regression_covar = torch.nn.ModuleList([
            gpytorch.kernels.ScaleKernel(regression_covar_base_class()),
            gpytorch.kernels.ScaleKernel(regression_covar_base_class()),
        ])
    else:
        regression_mean = regression_mean_base_class()
        regression_covar = gpytorch.kernels.ScaleKernel(
            regression_covar_base_class()
        )

    # Optimization-related
    # derivative
    optimization_acquisition_parameters = mfb.optimization.Acquisition_Parameters(
        best_f=-np.inf if optimization_maximize else np.inf,
        maximize=optimization_maximize,
    )

    if regression_type == "Sogpr":
        optimization_iterations = optimization_budget
        optimization_budget = None

        optimization_acquisition_class = getattr(
            mfb.machinelearning.acquisition_functions, optimization_acquisition_type
        )
        optimization_parameters_class = mfb.optimization.BayesianOptimizationTorch_Parameters
        optimizer_class = mfb.optimization.BayesianOptimizationTorch
    else:
        optimization_acquisition_class = mfb.machinelearning.acquisition_functions.VFUpperConfidenceBound
        optimization_parameters_class = mfb.optimization.MFBayesianOptimizationTorch_Parameters
        optimizer_class = mfb.optimization.MFBayesianOptimizationTorch

    #######################
    # Parameters: level 4 #
    #######################

    # Regression-related
    # derivative
    regression_parameters = regression_parameters_class(
        likelihood=regression_likelihood_class(),
        kernel=regression_covar,
        mean=regression_mean,
        # noise_fix=regression_noise_fix,
        opt_algo=regression_opt_class,
        opt_algo_kwargs=regression_opt_kwargs,
        training_iter=regression_training_iter,
    )

    # Optimization-related
    # derivative
    fidelity_functions = []

    for fidelity_parameter in [data_low_fidelity_parameter, data_high_fidelity_parameter]:
        # To be implemented accordingly
        if data_fidelity_parameter_name == "res":
            fidelity_function = CompasFunction(
                seed=123,
                jobnumber=jobnumber,
                slurm_jobid=slurm_jobid,
            )
        bounds = fidelity_function.scale_bounds
        domain = f3dasm.make_nd_continuous_domain(
            bounds=bounds, dimensionality=data_dimensionality
        )
        fidelity_functions.append(fidelity_function)

    multifidelity_function = mfb.functions.MultiFidelityFunction(
        fidelity_functions=fidelity_functions,
        fidelity_parameters=[0., 1.],
        costs=[optimization_lf_cost, 1.],
    )

    if regression_type == "Sogpr":
        optimization_function = fidelity_functions[1]
    else:
        optimization_function = multifidelity_function

    optimization_fidelity_initial_does = []
    for fidelity_parameter in [data_low_fidelity_parameter, data_high_fidelity_parameter]:
        samples_fidelity = f3dasm.ExperimentData(design=domain)

        # output_arr = data_initial_doe.output[
        #     data_initial_doe.input[data_fidelity_parameter_name] == fidelity_parameter
        # ][optimization_objective_name].values[:, None]

        # input_arr = data_initial_doe.input[
        #     data_initial_doe.input[data_fidelity_parameter_name] == fidelity_parameter
        # ].drop(data_fidelity_parameter_name, axis=1).values

        if 'rrotz' in data_initial_doe.input.columns:
            df_input = data_initial_doe.input.drop(
                columns=['rrotz']
            )
        else:
            df_input = data_initial_doe.input

        if data_initial_doe_size_hf is not None:
            output_arr = data_initial_doe.output.values[:data_initial_doe_size_hf]
            input_arr = df_input.values[:data_initial_doe_size_hf]
        else:
            output_arr = data_initial_doe.output.values
            input_arr = df_input.values

        # Scale the input data
        lower, upper = bounds.T
        input_arr = (input_arr - lower) / (upper - lower)

        samples_fidelity.add_numpy_arrays(input_arr, output_arr)
        optimization_fidelity_initial_does.append(samples_fidelity)

    if regression_type == "Sogpr":
        optimization_initial_doe = optimization_fidelity_initial_does[1]
    else:
        optimization_initial_doe = optimization_fidelity_initial_does

    #######################
    # Parameters: level 5 #
    #######################

    # Optimization-related
    # derivative
    optimization_parameters = optimization_parameters_class(
        regressor=regressor_class,
        acquisition=optimization_acquisition_class,
        regressor_hyperparameters=regression_parameters,
        acquisition_hyperparameters=optimization_acquisition_parameters,
    )

    #######################
    # Parameters: level 6 #
    #######################

    # Optimization-related
    # derivative
    optimizer = optimizer_class(
        data=None,  # Data set later
        hyperparameters=optimization_parameters.__dict__,
    )

    #######################
    # Optimization result #
    #######################

    optimization_result = mfb.run_optimization_with_doe(
        optimizer=optimizer,
        function=optimization_function,
        iterations=optimization_iterations,
        samples=optimization_initial_doe,
        budget=optimization_budget,
        jobnumber=jobnumber,
    )

    return optimization_result


class CompasFunction(f3dasm.Function):
    def __init__(
        self,
        seed=None,
        rrotz: float = 0.,
        jobnumber: int = 0,
        slurm_jobid=0,
        iteration_number: int = 0,
    ):
        super().__init__(seed)
        self.rrotz = rrotz
        self.jobnumber = jobnumber
        self.slurm_jobid = slurm_jobid
        self.iteration_number = iteration_number
        self.scale_bounds = np.array(
            [
                [15., 145.],
                [20., 145.],
                [5e-6, 12e-6],
                [20e-6, 37e-6],
            ]
        )

    def __call__(self, input_x: f3dasm.ExperimentData) -> np.ndarray:
        # input_x = np.insert(input_x, 1, self.res)
        logging.info('x (unscaled) at f3dasm.Function __call__ = %s' %
                     str(input_x))

        Objective = 0.
        while bool(Objective == 0.):
            output = compas_objective(
                ddx=input_x[0, 0],
                ddy=input_x[0, 1],
                CTE1=input_x[0, 2],
                CTE2=input_x[0, 3],
                rrotz=self.rrotz,
                jobnumber=self.jobnumber,
                iteration_number=self.iteration_number,
                slurm_jobid=self.slurm_jobid,
            )

            Objective = output

            if socket.gethostname() == 'hp':
                break

            if bool(Objective == 0.):
                logging.info('Null objective. Retrying with a random design.')
                input_x = np.random.rand(*input_x.shape)

        return np.array([Objective])[:, None]
