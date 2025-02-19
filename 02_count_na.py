# %% load modules

from pathlib import Path
import pandas as pd

# %%

FILE = Path("../data/clean/merged_smhs.csv")
d0 = pd.read_csv(FILE)
d0.shape
d0.columns

# %%
id_vars = ["district", "school", "teacher", "student"]
d1 = d0.melt(id_vars=id_vars)
n_na = (
    d1.groupby(["variable"])
    .apply(lambda x: x.isna().sum())[["value"]]
    .sort_values(["value"], ascending=False)
    .reset_index()
)
n_na.columns = ["variable", "n_na"]
n_na

n_na["prop_na"] = n_na["n_na"] / d0.shape[0]
n_na

# %%
d0[n_na.iloc[0, 0]].isna().sum()

# %%
n_na.to_csv("../data/clean/n_na.csv", index = False)
# %%
