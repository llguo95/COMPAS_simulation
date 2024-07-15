import f3dasm
import gpytorch
from matplotlib import pyplot as plt
import mfb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import torch

doe_data_path = Path(__file__).parent / "data" / "raw" / "doe_data_seed1.csv"
df_raw = pd.read_csv(doe_data_path, header=[0, 1], index_col=0)

df = df_raw.dropna().reset_index(drop=True)

df_rrotz0 = df[df.input.rrotz == 0].drop(
    columns=('input', 'rrotz')).reset_index(drop=True)

train_input, test_input = df_rrotz0.input.values[:-
                                                 10], df_rrotz0.input.values[-10:]
train_output, test_output = df_rrotz0.output.values[:-
                                                    10], df_rrotz0.output.values[-10:]


#######################
# Parameters: level 1 #
#######################

# Data-related
data_seed = 123
data_output_scaler_lf = StandardScaler()
data_output_scaler_hf = StandardScaler()

bounds = np.array(
    [
        [15., 145.],
        [20., 145.],
    ]
)

lower, upper = bounds.T

domain = f3dasm.make_nd_continuous_domain(
    bounds=bounds, dimensionality=2
)

# Regression-related
regression_training_iter = 100
regression_opt_kwargs = dict(lr=0.1)
regression_opt_class = torch.optim.Adam
regression_likelihood_class = gpytorch.likelihoods.GaussianLikelihood
regression_mean_base_class = gpytorch.means.ZeroMean
regression_covar_base_class = gpytorch.kernels.RBFKernel


def num_to_surrogate(regression_type, train_size_lf, train_size_hf):

    #######################
    # Parameters: level 2 #
    #######################

    # Regression-related
    regression_parameters_class = getattr(
        mfb.machinelearning.gpr, regression_type + '_Parameters'
    )
    regressor_class = getattr(mfb.machinelearning.gpr, regression_type)

    if regression_type in ["Cokgj", "Cokgd"]:
        if regression_type == "Cokgj":
            regression_noise_fix = 0

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
        regression_noise_fix = True
        regression_mean = regression_mean_base_class()
        regression_covar = gpytorch.kernels.ScaleKernel(
            regression_covar_base_class()
        )

    #######################
    # Parameters: level 3 #
    #######################

    # Regression-related
    regression_parameters = regression_parameters_class(
        likelihood=regression_likelihood_class(),
        kernel=regression_covar,
        mean=regression_mean,
        # noise_fix=regression_noise_fix,
        opt_algo=regression_opt_class,
        opt_algo_kwargs=regression_opt_kwargs,
        training_iter=regression_training_iter,
    )

    train_data = []

    for i, n in enumerate([train_size_lf, train_size_hf]):
        samples_fidelity = f3dasm.ExperimentData(design=domain)

        if i == 0:
            input_arr = train_input
            output_arr = train_output
        else:
            input_arr = train_input
            output_arr = train_output

        # Scale the input data
        input_arr_scaled = (input_arr - lower) / (upper - lower)

        # Scale the output data
        if i == 0:
            output_arr_scaled = data_output_scaler_lf.fit_transform(output_arr)
        else:
            output_arr_scaled = data_output_scaler_hf.fit_transform(output_arr)

        # Select random samples
        indices = np.random.choice(input_arr_scaled.shape[0], size=n)
        samples_fidelity.add_numpy_arrays(
            input_arr_scaled[indices], output_arr_scaled[indices])

        train_data.append(samples_fidelity)

    if regression_type == 'Sogpr':
        regression_train_data = train_data[1]
    else:
        regression_train_data = train_data

    #####################
    # Regression result #
    #####################

    regressor = regressor_class(
        train_data=regression_train_data,
        parameter=regression_parameters,
    )

    surrogate = regressor.train()

    return surrogate


input_hf = df_rrotz0.input.values
output_hf = df_rrotz0.output.values

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs=input_hf[:, 0], ys=input_hf[:, 1], zs=output_hf)
ax.set_xlabel('ddx')
ax.set_ylabel('ddy')
ax.set_zlabel('acc_nlcr')
ax.set_title('rrotz = 0')

ddx_plot_scaled = torch.linspace(0., 1., 50)
ddy_plot_scaled = torch.linspace(0., 1., 50)

ddx_plot = torch.linspace(15., 145., 50)
ddy_plot = torch.linspace(20., 145., 50)
plot_grid_list_unscaled = torch.meshgrid(ddx_plot, ddy_plot)

plot_grid_list = torch.meshgrid(ddx_plot_scaled, ddy_plot_scaled)
plot_grid = torch.stack(plot_grid_list)
plot_array = plot_grid.reshape(plot_grid.shape[0], -1).T

surrogate = num_to_surrogate(
    regression_type='Sogpr', train_size_lf=5, train_size_hf=90)
mean = data_output_scaler_hf.inverse_transform(
    surrogate.predict(plot_array).mean[:, None]).reshape(50, 50)

ax.plot_surface(
    plot_grid_list_unscaled[0], plot_grid_list_unscaled[1], mean, cmap='viridis', alpha=.5)
# plt.contourf(plot_grid_list_unscaled[0], plot_grid_list_unscaled[1], mean)
# plt.xlabel('ddx')
# plt.ylabel('ddy')
# cbar = plt.colorbar()
# cbar.set_label('acc_nlcr')
# plt.title('rrotz = 90')

plt.show()
