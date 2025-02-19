#%% load modules
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer,mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from sklearn.inspection import permutation_importance
import shap
import pickle
import numpy as np
import pandas as pd
import os



RANDOM_SEED = 42

#%%
def create_mean_column(data, pattern, new_col_name):
    col_names = list(data.filter(regex = pattern).columns)
    data[new_col_name] = data[col_names].mean(axis = 1)
    return data

def combine_mean_columns(data, col1, col2, new_col_name):
    data[new_col_name] = data[[col1, col2]].mean(axis= 1)
    return data

def create_mean_columns(data):
    create_mean_column(data, "^SB1", "Block_SB1")
    create_mean_column(data, "^SB2", "Block_SB2")
    create_mean_column(data, "^SB3", "Block_SB3")
    create_mean_column(data, "^SB4", "Block_SB4")
    create_mean_column(data, "^SB5", "Block_SB5")
    create_mean_column(data, "^SB6", "Block_SB6")
    create_mean_column(data, "^SB7", "Block_SB7")
    create_mean_column(data, "^SC1", "Block_SC1")
    create_mean_column(data, "^SC2", "Block_SC2")
    create_mean_column(data, "^SC3", "Block_SC3")
    create_mean_column(data, "^SD3", "Block_SD3")
    create_mean_column(data, "^SD4", "Block_SD4")
    create_mean_column(data, "^SE1", "Block_SE1")
    create_mean_column(data, "^SE3", "Block_SE3")
    create_mean_column(data, "^SE5", "Block_SE5")
    create_mean_column(data, "^SE6", "Block_SE6")
    create_mean_column(data, "^SF2", "Block_SF2")

    create_mean_column(data, "^TSSA1", "Block_TA1")
    create_mean_column(data, "^TSSB1", "Block_TB1")
    create_mean_column(data, "^TSSB4", "Block_TB4")
    create_mean_column(data, "^TSSB5", "Block_TB5")
    create_mean_column(data, "^TSSB6", "Block_TB6")
    create_mean_column(data, "^TSSB7", "Block_TB7")
    create_mean_column(data, "^TSSC1", "Block_TC1")
    create_mean_column(data, "^TSSD1", "Block_TD1")
    create_mean_column(data, "^TSSD2", "Block_TD2")
    create_mean_column(data, "^TSSD3", "Block_TD3")
    create_mean_column(data, "^TSSE1", "Block_TE1")
    create_mean_column(data, "^TSSF1", "Block_TF1")
    create_mean_column(data, "^TSSG1", "Block_TG1")
    create_mean_column(data, "^TSSH1", "Block_TH1")
    create_mean_column(data, "^TSSI4", "Block_TI4")

    create_mean_column(data, "^PA1", "Block_PA1")
    create_mean_column(data, "^PB1", "Block_PB1")
    create_mean_column(data, "^PB2", "Block_PB2")
    create_mean_column(data, "^PB3", "Block_PB3")
    create_mean_column(data, "^PB4", "Block_PB4")
    create_mean_column(data, "^PC1", "Block_PC1")
    create_mean_column(data, "^PC2", "Block_PC2")
    create_mean_column(data, "^PD1", "Block_PD1")
    create_mean_column(data, "^PD3", "Block_PD3")
    create_mean_column(data, "^PD4", "Block_PD4")
    create_mean_column(data, "^PD5", "Block_PD5")
    create_mean_column(data, "^PE1", "Block_PE1")
    create_mean_column(data, "^PF1", "Block_PF1")
    create_mean_column(data, "^PG3", "Block_PG3")

    data["Block_SD1"] = data[["SD11", "SD12", "SD13", "SD14", "SD15",
                        "SD16", "SD17"]].mean(axis= 1)
    data["Block_SD8"] = data[["SD8Smoke", "SD9Marij", "SD10Alc"]].mean(axis= 1)

    combine_mean_columns(data, 'Block_SD8', 'Block_SD4', 'Block_SD4SD8') # lifestyle
    combine_mean_columns(data, 'Block_TD2', 'Block_TD3', 'Block_TD2TD3') # Teaching pratices and interactions
    data = data.drop(['Block_SD4', 'Block_SD8','Block_TD2', 'Block_TD3'], axis= 1)

    block_data = data.filter(regex= "^Block_")

    return block_data

# define outcomes
def group_outcomes(data):
    data["Positive_mental_health_outcome"] = data.filter(regex= "^SD2").mean(axis= 1)
    data["Mental_problems_outcome"] = data.filter(regex= "^SD5").mean(axis= 1)
    data["ADHD_outcome"] = data[["SD51","SD52",
                    "SD53","SD54"]].mean(axis= 1)
    data["ODD_outcome"] = data[["SD55","SD56", "SD57",
                "SD58", "SD59"]].mean(axis= 1)
    data["CD_outcome"] = data[["SD519","SD520", "SD521","SD522", "SD523", 
                "SD524", "SD525", "SD526"]].mean(axis= 1)
    data["Externalizing_outcome"] = data[["ADHD_outcome", "ODD_outcome",
                                           "CD_outcome"]].mean(axis= 1)
    data["Depression_outcome"] = data[["SD510", "SD511", "SD512", 
                        "SD513", "SD514"]].mean(axis= 1)
    data["Anxiety_outcome"] = data[["SD515", "SD516", "SD517",
                        "SD518"]].mean(axis= 1)
    data["Internalizing_outcome"] = data[["Depression_outcome",
                                           "Anxiety_outcome"]].mean(axis= 1)
  
    outcome_data = data.filter(regex = "_outcome$")
    outcome_data = outcome_data.drop(columns= ["ADHD_outcome","ODD_outcome","CD_outcome",
                    "Depression_outcome","Anxiety_outcome"], axis= 1)                      
    outcome_list = list(outcome_data.filter(regex = "_outcome$").columns)
    return  outcome_data, outcome_list
# remove rows where the target outcome column has missing values
def get_train_test_data(train_df, test_df, feature_labels, target_outcome):
    train_df.dropna(subset=[target_outcome], inplace= True)
    test_df.dropna(subset=[target_outcome], inplace= True)
    X_train = train_df[feature_labels]
    y_train = train_df[target_outcome]
    ID_train = train_df['cluster']
    X_test= test_df[feature_labels]
    y_test = test_df[target_outcome]
    ID_test = test_df['cluster']
    return X_train, y_train, ID_train, X_test, y_test, ID_test

#%%
#  define ridge model function
def evaluate_ridge(X_train, y_train, ID_train, X_test, y_test, ID_test):
    param_distributions = {'ridge__alpha': [1e3, 1e5, 1e6, 1e7, 1e8]}

    # Create a pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])
    
    # Define scoring metrics
    scoring = {'MAE': 'neg_mean_absolute_error',
               'MSE': 'neg_mean_squared_error'}
    
    # Define RandomizedSearchCV
    ridge_model = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=param_distributions, 
        n_iter=200, 
        scoring= scoring,
        refit='MAE', 
        cv=gss.split(X_train, groups=ID_train), 
        verbose=2, 
        random_state=RANDOM_SEED,
        n_jobs=1
    )

    #fit model
    ridge_model.fit(X_train, y_train)

    # predict 
    ridge_train_pred = ridge_model.predict(X_train)
    ridge_test_pred = ridge_model.predict(X_test)

    # access the best parameters and the best estimator
    best_alpha = ridge_model.best_params_['ridge__alpha']
    best_ridge_model = ridge_model.best_estimator_ 
    access_ridge_model = best_ridge_model.named_steps["ridge"]

    # Cross-validation for MAE, R2, and MSE
    cv_mae = -ridge_model.cv_results_['mean_test_MAE'][ridge_model.best_index_]
    cv_mse = -ridge_model.cv_results_['mean_test_MSE'][ridge_model.best_index_]


    metric_values = {
        "train_mae": round(mean_absolute_error(y_train, ridge_train_pred), 4),
        "train_mse": round(mean_squared_error(y_train, ridge_train_pred), 4),
        
        "cv_mae": round(cv_mae.mean(), 4),
        "cv_mse": round(cv_mse.mean(), 4),


 
        "test_mae": round(mean_absolute_error(y_test, ridge_test_pred), 4),
        "test_mse": round(mean_squared_error(y_test, ridge_test_pred), 4)

    }
    return best_ridge_model, access_ridge_model,best_alpha, metric_values 

# lasso regression
def evaluate_lasso(X_train, y_train, ID_train, X_test, y_test, ID_test):

    param_distributions = {'lasso__alpha': np.logspace(-4, 4, 20)}

    # Create a pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('lasso', Lasso(selection='random'))
    ])
    
    # Define scoring metrics
    scoring = {'MAE': 'neg_mean_absolute_error',
               'MSE': 'neg_mean_squared_error'
               }
    
    # Define RandomizedSearchCV
    lasso_model = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=param_distributions, 
        n_iter=200, 
        scoring= scoring,
        refit='MAE', 
        cv=gss.split(X_train, groups=ID_train), 
        verbose=2, 
        random_state=RANDOM_SEED,
        n_jobs=1
    )

    lasso_model.fit(X_train, y_train)

    lasso_train_pred = lasso_model.predict(X_train)
    lasso_test_pred = lasso_model.predict(X_test)

    # access the best parameters and the best estimator
    best_alpha = lasso_model.best_params_['lasso__alpha']
    best_lasso_model = lasso_model.best_estimator_ #give the complete pipeline
    access_lasso_model = best_lasso_model.named_steps["lasso"] 
    

    # Cross-validation metrics
    cv_mae = -lasso_model.cv_results_['mean_test_MAE'][lasso_model.best_index_]
    cv_mse = -lasso_model.cv_results_['mean_test_MSE'][lasso_model.best_index_]


    metric_values = {
    
    "train_mae": round(mean_absolute_error(y_train, lasso_train_pred), 4),
    "train_mse": round(mean_squared_error(y_train, lasso_train_pred), 4),

    "cv_mae": round(cv_mae.mean(), 4),
    "cv_mse": round(cv_mse.mean(), 4),

    "test_mae": round(mean_absolute_error(y_test, lasso_test_pred), 4),
    "test_mse": round(mean_squared_error(y_test, lasso_test_pred), 4)

    }

    return best_lasso_model, access_lasso_model,best_alpha, metric_values

# lightgbm_gain
def evaluate_lgbm_gain(X_train, y_train, ID_train, X_test, y_test, ID_test):
    # define random_grid
    n_estimators = [int(x) for x in np.linspace(start= 500, stop= 1000, num= 100)]
    num_leaves = [50, 55, 60, 65 ]
    max_depth = [1,2,3,4, 5]  
    learning_rate = [0.02, 0.03, 0.04, 0.05]
    min_child_weight = [2,4, 6, 8, 10]
    min_child_samples = [1, 2, 3, 4]

    random_grid = {
        'lgbm__n_estimators': n_estimators,
        'lgbm__num_leaves': num_leaves,
        'lgbm__max_depth': max_depth,
        'lgbm__learning_rate': learning_rate,
        'lgbm__min_child_weight': min_child_weight,
        'lgbm__min_child_samples': min_child_samples,
    }

    # define scoring 
    scoring = {
    'MAE': make_scorer(mean_absolute_error, greater_is_better= False),
    'MSE': make_scorer(mean_squared_error, greater_is_better= False)
    }

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lgbm', LGBMRegressor(importance_type="gain"))
        ])

    # Define RandomizedSearchCV
    lgbm_model = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=random_grid, 
        n_iter=200, 
        scoring= scoring,
        refit='MAE',
        cv=gss.split(X_train, groups=ID_train), 
        verbose=2, 
        random_state=RANDOM_SEED,
        n_jobs=1
    )

    # Fit the random search model
    lgbm_model.fit(X_train, y_train)

    # access the best parameters and the best estimator
    best_param = lgbm_model.best_params_
    best_lgbm_model = lgbm_model.best_estimator_
    lightgbm_model = best_lgbm_model.named_steps["lgbm"]

    # Cross-validation metrics
    cv_mae = -lgbm_model.cv_results_['mean_test_MAE'][lgbm_model.best_index_]
    cv_mse = -lgbm_model.cv_results_['mean_test_MSE'][lgbm_model.best_index_]


    #  use best_lgbm_model for predictions,
    lgbm_train_pred = best_lgbm_model.predict(X_train)
    lgbm_test_pred = best_lgbm_model.predict(X_test)

    # Metrics dictionary
    metric_values = {
        "train_mae": round(mean_absolute_error(y_train, lgbm_train_pred), 4),
        "train_mse": round(mean_squared_error(y_train, lgbm_train_pred), 4),

        "cv_mae": round(cv_mae.mean(), 4),
        "cv_mse": round(cv_mse.mean(), 4),



        "test_mae": round(mean_absolute_error(y_test, lgbm_test_pred), 4),
        "test_mse": round(mean_squared_error(y_test, lgbm_test_pred), 4)

    }

    return best_lgbm_model, lightgbm_model, best_param, metric_values


#%%
#%%  feature importance analysis


# lightgbm_gain feature importance
def lgbm_gain_feature_importance(immigration, X_train, model_name, target_outcome, count): 
    save_path = f'../results/imm_feature_imp/{immigration}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gain_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model_name.feature_importances_
    })
      
    gain_importance= gain_importance.sort_values(by="Importance", ascending=False)
    gain_importance.to_csv(f"{save_path}/lgbm_{immigration}_{target_outcome}{count}.csv", index= False)

# permutation lgbm 
def permutation_feature_importance(immigration, X_train, X_test, y_test, model_name, target_outcome, count):
    save_path = f'../results/imm_feature_imp/{immigration}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    perm_importance = permutation_importance(model_name, X_test, y_test, 
                                        n_repeats=30, random_state= RANDOM_SEED, 
                                        n_jobs=1, scoring= 'r2')

    perm_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': perm_importance.importances_mean,
    })
    
    perm_importance_df_sorted = perm_importance_df.sort_values(by="Importance", ascending=False)

    perm_importance_df_sorted.to_csv(f"{save_path}/permutation_{immigration}_{target_outcome}{count}.csv", index= False)

#lgbm gain shapvic
def shap_feature_importance(immigration, X_train, X_test, pipeline, target_outcome, model_type, count):
    save_path = f'../results/imm_feature_imp/{immigration}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if model_type == 'lgbm':
        transformed_X = pipeline[:-1].transform(X_train)
        masker = shap.maskers.Independent(data=transformed_X, max_samples=800)
        explainer = shap.Explainer(pipeline[-1], 
                                    n_samples=transformed_X.shape[0],
                                    masker=masker)
        shap_values = explainer.shap_values(pipeline[:-1].transform(X_test),
                                            check_additivity=False)

    elif model_type in ['ridge','lasso']:
        def f(data):
            return pipeline.predict(data)
        masker = shap.maskers.Independent(data=X_train, max_samples=800)
        explainer = shap.KernelExplainer(f, shap.kmeans(X_train, 80))
        shap_values = explainer.shap_values(X_test, nsamples=500)
        
    shap_sum = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': shap_sum,
    })


    importance_df.sort_values(by='Importance', ascending=True, inplace=True)
    importance_df.to_csv(f"{save_path}/shap_{immigration}_{target_outcome}{count}.csv", index= False)

#%%
FILE = Path("../data/clean/untouched_dataset.csv")
d0 = pd.read_csv(FILE)
d0 = d0.drop(columns= ["SF4DadCan","SF3MomCan"], axis=1)
# group shuffle split for training and testing splits; 
# using it here to get train and test IDs
gss = GroupShuffleSplit(n_splits= 1, train_size= 0.8, random_state= RANDOM_SEED) 
gss.get_n_splits()
 	
# Generate indices to split data into training and test set.
for train_index, test_index in gss.split(d0, groups=d0["cluster"]):
    train_IDs = d0.iloc[train_index]["cluster"]
    test_IDs = d0.iloc[test_index]["cluster"]

# group shuffle splits for cross-validation
gss = GroupShuffleSplit(n_splits= 5, train_size= 0.8, random_state= RANDOM_SEED)

#%% load data
def ml_featureimportance(immigration, train_IDs, test_IDs):
    for count in range(10):
        FILE = Path(f"../data/clean/imputed_train/imputed_train{count}.csv")
        train_df = pd.read_csv(FILE)
        
        FILE = Path(f"../data/clean/imputed_test/imputed_test{count}.csv")
        test_df = pd.read_csv(FILE) 

        if immigration == "imm":
            train_df = train_df[(train_df['SF3MomCan'] == 0) | (train_df['SF4DadCan'] == 0) ]
            train_df = train_df.drop(columns= ['SF3MomCan','SF4DadCan'], axis= 1)
            test_df = test_df[(test_df['SF3MomCan'] == 0) | (test_df['SF4DadCan'] == 0)]
            test_df = test_df.drop(columns= ['SF3MomCan','SF4DadCan'], axis= 1)
        elif immigration == "nonimm":
            train_df = train_df[(train_df['SF3MomCan'] == 1) | (train_df['SF4DadCan'] == 1) ]
            train_df = train_df.drop(columns= ['SF3MomCan','SF4DadCan'], axis= 1)
            test_df = test_df[(test_df['SF3MomCan'] == 1) | (test_df['SF4DadCan'] == 1)]
            test_df = test_df.drop(columns= ['SF3MomCan','SF4DadCan'], axis= 1)

        # predictors
        demographics_cols = [
            "SA1Sex",
            "SA2Age", 
            "SA3Grade",
            "SA4Canadaborn",
            "SF5ParentEd",
            "TSSI1Sex", 
            "TSSI2Canadaborn",
            "PGSex"
        ]
        demographics_cols.extend(list(train_df.filter(regex= \
                                                "^SA6|^SF1|^TSSI3|^PSCH|^TSSB2").columns))


        demographics_df_train = train_df[demographics_cols].reset_index(drop= True)
        demographics_df_test = test_df[demographics_cols].reset_index(drop= True)

        # apply functions 
        block_train = create_mean_columns(train_df)
        block_test = create_mean_columns(test_df)
        outcome_train, outcome_list = group_outcomes(train_df)
        outcome_test, outcome_list = group_outcomes(test_df)

        # update train and test datasets so that it contains desired variables for analysis
        block_train = block_train.reset_index(drop= True)
        outcome_train = outcome_train.reset_index(drop= True)
        block_test = block_test.reset_index(drop= True)
        outcome_test = outcome_test.reset_index(drop= True)
        train_IDs = train_IDs.reset_index(drop= True)
        test_IDs = test_IDs.reset_index(drop= True)
        train_df = pd.concat([demographics_df_train, block_train, outcome_train, train_IDs], axis= 1)
        test_df =  pd.concat([demographics_df_test, block_test, outcome_test, test_IDs], axis= 1)

        print(train_df.shape[0], test_df.shape[0])
        # feature selection
        demographics_cols.remove('SA3Grade')
        demographics_cols.remove('PSCH_level')
        block_train_cols = block_train.columns.tolist() 
        feature_labels = demographics_cols + block_train_cols
        print(len(feature_labels)) #87

        train_df = train_df.drop(columns=['SA3Grade', 'PSCH_level'], axis=1)
        train_df = test_df.drop(columns=['SA3Grade', 'PSCH_level'], axis=1)

        # LOOP
        filename = f'../results/imm_model_results/{immigration}/{immigration}_model_results{count}.pkl'

        if not os.path.exists(f'../results/imm_model_results/{immigration}'):
            os.mkdir(f'../results/imm_model_results/{immigration}')

        if os.path.exists(filename):
            results_dict = pd.read_pickle(filename)
        else:
            results_dict = {}

        for outcome_name in outcome_list:

            metrics = {}

            X_train, y_train, ID_train, X_test, y_test, ID_test = get_train_test_data(train_df, test_df, feature_labels, outcome_name)

            #try:
            ridge_pipeline, ridge_model, ridge_best_alpha, ridge_metrics = evaluate_ridge(X_train, y_train, ID_train, X_test, y_test, ID_test)
            #except ValueError:
            #    return X_train, y_train, X_test, y_test

            results_dict[outcome_name] = {'ridge':{'metrics':ridge_metrics,
                                                    'best_params':ridge_best_alpha,
                                                    'pipeline':ridge_pipeline
                                                    }
                                        }
            metrics['ridge'] = ridge_metrics['test_mae']

                                                                                
            lasso_pipeline, lasso_model, lasso_best_alpha, lasso_metrics = evaluate_lasso(X_train, y_train, ID_train, X_test, y_test, ID_test)

            results_dict[outcome_name]['lasso'] ={'metrics':lasso_metrics,
                                                'best_params':lasso_best_alpha,
                                                'pipeline':lasso_pipeline
                                                }

            metrics['lasso'] = lasso_metrics['test_mae']


            lgbm_pipeline, lgbm_model, lgbm_best_param, lgbm_metrics = evaluate_lgbm_gain(X_train, y_train, ID_train, X_test, y_test, ID_test) 

            results_dict[outcome_name]['lgbm'] = {'metrics':lgbm_metrics,
                                            'best_params':lgbm_best_param,
                                            'pipeline':lgbm_pipeline,
                                            'model':lgbm_model
                                            }

            metrics['lgbm'] = lgbm_metrics['test_mae']

            best_model_type = min(metrics.items(), key=lambda x: x[1])[0]

            with open(filename, 'wb') as file:
                pickle.dump(results_dict, file)


            if best_model_type == "lgbm":
                lgbm_gain_feature_importance(immigration, X_train, lgbm_model, outcome_name, count)
            
            permutation_feature_importance(immigration, X_train, X_test, y_test,results_dict[outcome_name][best_model_type]['pipeline'], outcome_name, count)

            shap_feature_importance(immigration, X_train, X_test, results_dict[outcome_name][best_model_type]['pipeline'], outcome_name, best_model_type,count)

# %%
imm_group = ["imm", "nonimm"]

for immigration in imm_group[1:]:
        ml_featureimportance(immigration, train_IDs, test_IDs)
#%%