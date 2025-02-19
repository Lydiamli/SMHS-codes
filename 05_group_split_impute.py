# %% load modules
import miceforest as mf
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from miceforest import mean_match_default
#%%
pd.set_option(
    "display.max_rows",
    8,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.expand_frame_repr",
    True,
    "display.max_colwidth",
    None,
)

OPTIMIZATION_STEPS = 5
RANDOM_SEED = 42
ITERATION_COUNT = 20
# 10 train datasets
DATASET_COUNT = 10
# %% load data
FILE = Path("../data/clean/untouched_dataset.csv")
d0 = pd.read_csv(FILE)
d0["ID"] = d0["district"].astype(str)+d0["school"].astype(str)+ \
    d0["teacher"].astype(str)

#%% stratified split based on district-school-teacher combination
gss = GroupShuffleSplit(n_splits= 1, train_size= 0.8, random_state= RANDOM_SEED) 
gss.get_n_splits()
 	
# Generate indices to split data into training and test set.
for train_index, test_index in gss.split(d0, groups=d0["ID"]):
    train_df = d0.iloc[train_index]
    test_df = d0.iloc[test_index]

#%% remove outcome from imputation and specify categotical variables
train_df.drop("ID", axis =1, inplace= True)
outcome_cols = train_df.filter(regex= "^SD2|^SD5").columns.values
train_outcome_df = train_df[outcome_cols]
train_df.drop(outcome_cols, axis = 1, inplace = True)
id_vars = ["district", "school", "teacher"]
train_df[id_vars] = train_df[id_vars].astype('category')
#%% mutiple imputations
scheme_mmc = mean_match_default.copy()
scheme_mmc.set_mean_match_candidates(0)

# create a folder 
save_dir = '../data/clean/group_imputed_train'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



# create kernel
kernel = mf.ImputationKernel(
  train_df,
  datasets = DATASET_COUNT,
  mean_match_scheme = scheme_mmc,
  save_all_iterations = True,
  random_state= RANDOM_SEED,
)

# find optimal lightgbm parameters
optimal_parameters, losses = kernel.tune_parameters(
    dataset= 0, 
    optimization_steps = OPTIMIZATION_STEPS, 
    random_state = RANDOM_SEED
)

# impute on predictors 
kernel.mice(iterations= ITERATION_COUNT, 
            variable_parameters=optimal_parameters)

# save imputed train datasets
for i in range(0, DATASET_COUNT):
    completed_dataset = kernel.complete_data(dataset = i, 
                                             iteration= ITERATION_COUNT)
    completed_dataset = pd.concat([train_outcome_df, completed_dataset], axis = 1)
    completed_dataset.to_csv(f"{save_dir}/imputed_train{i}.csv", index=False)
    


#%% mean covergence correaltion for each variable in train dataset
all_correlations_per_var_per_dataset = {}
mean_corr_per_variable_across_dataset= {}

for count, variable_name in enumerate(train_df.columns.values):
    if not(count in kernel.unimputed_variables) \
        and (count not in kernel.categorical_variables): 
        _, _, correlations = kernel.plot_correlations(variables = [variable_name])
        mean_corr_per_variable_across_dataset[variable_name] = \
            np.mean(correlations[count][ITERATION_COUNT])
        all_correlations_per_var_per_dataset[variable_name] = \
            correlations[count][ITERATION_COUNT]

df_mean_correlation = pd.DataFrame(mean_corr_per_variable_across_dataset.values(), 
                                   index = mean_corr_per_variable_across_dataset.keys(),
                                   columns= ["corr_column"])
df_mean_correlation["variable"] = df_mean_correlation.index
df_mean_correlation.to_csv(f"{save_dir}/mean_convergence_impution_correlations.csv", index= False)
    

#%% impute on predictors in test dataset
# remove outcome from imputation and specify categotical variables
test_df.drop("ID", axis =1, inplace= True)
outcome_cols = test_df.filter(regex= "^SD2|^SD5").columns.values
test_outcome_df = test_df[outcome_cols]
test_df.drop(outcome_cols, axis = 1, inplace = True)
id_vars = ["district", "school", "teacher"]
test_df[id_vars] = test_df[id_vars].astype('category')

#create folder
save_dir = '../data/clean/group_imputed_test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# compile candidate predictions
kernel.compile_candidate_preds()

# mutiple imputations
scheme_mmc = mean_match_default.copy()
scheme_mmc.set_mean_match_candidates(0)


# 100 test datasets
DATASET_COUNT = 100

# create kernel
kernel = mf.ImputationKernel(
  test_df,
  datasets = DATASET_COUNT,
  mean_match_scheme = scheme_mmc,
  save_all_iterations = True,
  random_state= RANDOM_SEED,
)

# find optimal lightgbm parameters
optimal_parameters, losses = kernel.tune_parameters(
    dataset= 0, 
    optimization_steps = OPTIMIZATION_STEPS, 
    random_state = RANDOM_SEED
)

# impute on predictors 
kernel.mice(iterations= ITERATION_COUNT, 
            variable_parameters=optimal_parameters)

# save imputed train datasets
for i in range(0, DATASET_COUNT):
    completed_dataset = kernel.complete_data(dataset = i, 
                                             iteration= ITERATION_COUNT)
    completed_dataset = pd.concat([test_outcome_df, completed_dataset], axis = 1)
    completed_dataset.to_csv(f"{save_dir}/imputed_test{i}.csv", index=False)
    


# %%

