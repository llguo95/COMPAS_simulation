from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

doe_data_path = Path(__file__).parent / "doe_data.csv"
df_raw = pd.read_csv(doe_data_path, header=[0, 1], index_col=0)

df = df_raw.dropna().reset_index(drop=True)

df_rrotz0 = df[df.input.rrotz == 0].drop(
    columns=('input', 'rrotz')).reset_index(drop=True)

input_hf = df_rrotz0.input.values
output_hf = df_rrotz0.output.values

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs=input_hf[:, 0], ys=input_hf[:, 1], zs=output_hf)
ax.set_xlabel('ddx')
ax.set_ylabel('ddy')
ax.set_zlabel('acc_nlcr')

plt.show()
