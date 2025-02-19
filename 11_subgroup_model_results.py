# %% load modules
import pandas as pd
import os
import pickle
os.getcwd()

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

#%%
def model_fit_results(mental_health, gender, model, metrics_df): 

    train_mae_values = []
    train_mse_values = []

    cv_mae_values = []
    cv_mse_values = []

    test_mae_values = []
    test_mse_values = []


    # Loop over the 10 files
    for x in range(10):
        file_path = f"../results/gender_model_results/{gender}/{gender}_model_results{x}.pkl"
        try:
            with open(file_path, 'rb') as file:
            # Load the data from the file
                data = pickle.load(file)

                # Extract metrics and add to the list if they exist
                try:
                    train_mae = data[mental_health][model]['metrics']['train_mae']
                    train_mae_values.append(train_mae)
                except KeyError:
                    print(f"train_mae not found in file {file_path}")

                try:
                    train_mse = data[mental_health][model]['metrics']['train_mse']
                    train_mse_values.append(train_mse)
                except KeyError:
                    print(f"train_mse not found in file {file_path}")

                try:
                    cv_mae = data[mental_health][model]['metrics']['cv_mae']
                    cv_mae_values.append(cv_mae)
                except KeyError:
                    print(f"cv_mae not found in file {file_path}")

                try:
                    cv_mse = data[mental_health][model]['metrics']['cv_mse']
                    cv_mse_values.append(cv_mse)
                except KeyError:
                    print(f"cv_mse not found in file {file_path}")
                

                try:
                    test_mae = data[mental_health][model]['metrics']['test_mae']
                    test_mae_values.append(test_mae)
                except KeyError:
                    print(f"test_mae not found in file {file_path}")

                try:
                    test_mse = data[mental_health][model]['metrics']['test_mse']
                    test_mse_values.append(test_mse)
                except KeyError:
                    print(f"test_mse not found in file {file_path}")


        except FileNotFoundError:
            print(f"File {file_path} not found.")


        # Calculate the mean and standard deviation of the metrics if values exist
        mean_train_mae = round(sum(train_mae_values) / len(train_mae_values), 4) if train_mae_values else None
        mean_train_mse = round(sum(train_mse_values) / len(train_mse_values), 4) if train_mse_values else None

        mean_cv_mae = round(sum(cv_mae_values) / len(cv_mae_values), 4) if cv_mae_values else None
        mean_cv_mse = round(sum(cv_mse_values) / len(cv_mse_values), 4) if cv_mse_values else None


        mean_test_mae = round(sum(test_mae_values) / len(test_mae_values), 4) if test_mae_values else None
        mean_test_mse = round(sum(test_mse_values) / len(test_mse_values), 4) if test_mse_values else None
  

        # Append the metrics to the DataFrame
        metrics = {
            'mental_health': mental_health, 'model': model, 
            'mean_train_mae': mean_train_mae, 
            'mean_train_mse': mean_train_mse, 
            'mean_cv_mae': mean_cv_mae, 
            'mean_cv_mse': mean_cv_mse, 
            'mean_test_mae': mean_test_mae, 
            'mean_test_mse': mean_test_mse,
            'gender': gender
        }

        # Convert the metrics dictionary to a DataFrame
        metrics_df_new = pd.DataFrame([metrics])

        # Concatenate the new metrics DataFrame with the existing one
        metrics_df = pd.concat([metrics_df, metrics_df_new], ignore_index=True)
        return metrics_df


#%%
# Initialize the metrics DataFrame 
metrics_df = pd.DataFrame(columns=[
    'mental_health', 
    'model', 
    'gender',

    'mean_train_mae', 
    'mean_train_mse', 

    'mean_cv_mae', 
    'mean_cv_mse', 

    'mean_test_mae', 
    'mean_test_mse', 

])
# Define mental health outcomes and models
mental_health_outcomes = [
    'Mental_problems_outcome', 
    'Positive_mental_health_outcome',
    'Externalizing_outcome',
    'Internalizing_outcome',
]
models = ['ridge', 'lasso', 'lgbm']

genders = ["male","female"]
# Apply the function to all combinations
for outcome in mental_health_outcomes:
    for model in models:
        for gender in genders:
            metrics_df = model_fit_results(outcome, gender, model, metrics_df)

# Save the metrics DataFrame to a CSV file
metrics_df.to_csv("../results/gender_model_metrics_results.csv", index = False)

metrics_df

# repreat the same steps for imm, edu files
# %%
