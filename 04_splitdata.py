# %% load modules
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 0

# %% load data
FILE = Path("../data/raw/encoded.csv")
d0 = pd.read_csv(FILE)
d0.shape

#%% 
pilot_df, untouched_df = train_test_split(d0, test_size = 0.8, random_state = RANDOM_SEED)
pilot_df.to_csv("../data/clean/pilot_dataset.csv", index = False) 
untouched_df.to_csv("../data/clean/untouched_dataset.csv", index = False) # this is the data used for the manuscript
# %%
