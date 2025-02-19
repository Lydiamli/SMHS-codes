rm(list = ls())
library(data.table)
library(glue)
library(tidyverse)
library(grf)

mapping_vector <- c(
    SA1Sex = "Gender",
    SA2Age = "Age",
    SA4Canadaborn = "Born in Canada",
    SA6SA = "Self-identified as South Asian",
    SA6White = "Self-identified as White",
    SA6WA = "Self-identified as West Asian/Arab",
    SF1Alone = "Live alone",
    SF1BioMother = "Spend most time with biological mother at home",
    SF1NonBiofather = "Spend most time with non-biological father at home",
    SF1Other = "Spend most time with other(s) at home",
    SF3MomCan = "Mother born in Canada",
    SF4DadCan = "Father born in Canada",
    SF5ParentEd = "Parental education",
    Block_SB1 = "School climate",
    Block_SB2 = "School belonging",
    Block_SB3 = "Perceived school safety",
    Block_SB4 = "Academic achievement",
    Block_SB5 = "School extracirricular activities",
    Block_SB6 = "Peer bullying",
    Block_SB7 = "School behavioral infractions",
    Block_SC1 = "Teacher-student interaction",
    Block_SC2 = "Student group interaction",
    Block_SC3 = "Class preparedness",
    Block_SD1 = "Social competence",
    Block_SD4SD8 = "Lifestyle(eg,exercise,\nsleep, substance use)",
    Block_SD3 = "Friendship quality",
    Block_SE1 = "Received school mental health service",
    Block_SE3 = "Help-seeking willingness",
    Block_SE5 = "Professional help-seeking",
    Block_SE6 = "Informal help-seeking",
    Block_SF2 = "Family relationship quality",


    Block_PB1 = "(P)Social and emotional learning programs",
    Block_PB2 = "(P)Violence prevention or peace promotion programs",
    Block_PB3 = "(P)Risk prevention or health promotion",
    Block_PC1 = "(P)Mental health service coordination",
    Block_PD3 = "(P)School mental health services availability",
    Block_PD4 = "(P)Referral and coordination practice",
    Block_PD5 = "(P)Resources and practices for mental health emergencies",

)

# Function to merge feature importance data####
import_and_merge_features <- function(gender, mental_health) {
    methods <- c("shap", "permutation", "lgbm")
    dataframes <- list()
    
    # Loop through each method and import data
    for (method in methods) {
        file_path <- glue("../results/gender_feature_imp/{gender}/average_{gender}_{method}_{mental_health}.csv")
        
        if (file.exists(file_path)) {
            df <- fread(file_path)
            df <- df[order(imp_mean, decreasing = TRUE)][1:10]
            dataframes[[method]] <- df 
        } else {
            cat(glue("File {file_path} does not exist for {method}. Skipping.\n"))
        }
    }
    
    combined <- Reduce(function(x, y) merge(x, y, by = "Feature", all = FALSE), dataframes)
    setDT(combined)
    return(unique(combined$Feature))
}


# Function for evaluating grf and saving results####
grf_top_features <- function(gender, mental_health){
    
    female_features <- import_and_merge_features("female", mental_health)
    male_features <- import_and_merge_features("male", mental_health)
    
    # Combine the features for 'female' and 'male'
    top_features <- union(female_features, male_features)
    d0 <- fread("../data/clean/imputed_processed/complete0.csv") ## due to computational cost, in GRF analysis, we combined one imputed train data withone imputed test data to generate a complete imputed dataset
    selected_cols <- c("SA1Sex", "cluster", mental_health, top_features)
    d1 <- d0[, ..selected_cols, with = FALSE]
    d1 <- na.omit(d1, cols = mental_health)
    outcome <- d1[[mental_health]]
    subgroup <- d1[["SA1Sex"]]
    cluster <- d1[["cluster"]]
    cluster_numeric <- as.numeric(as.factor(cluster))
    formula_string <- paste("~ 0 +", paste(top_features, collapse =" + "))
    formula <- as.formula(formula_string)
    X <- model.matrix(formula, d1)
    
    # fit model ####
    set.seed(123)
    cf <- causal_forest(
        X,  outcome, subgroup,
        tune.parameters = "all",
        clusters = cluster_numeric
        
    )
    
    directory_path <- "../results/subgroup_results/gender"
    if (!dir.exists(directory_path)) {
        dir.create(directory_path, recursive = TRUE)
    }
    
    
    for (colname in colnames(X)) {
        female_features <- sapply(female_features, function(x) ifelse(x == colname, mapping_vector[colname], x))
        male_features <- sapply(male_features, function(x) ifelse(x == colname, mapping_vector[colname], x))
        
    }
    
    importance_scores <- variable_importance(cf)
    names(importance_scores) <- colnames(X)
    importance_scores <- sort(importance_scores, decreasing = TRUE)
    
    
    model_results <- list(
        female_features = female_features,
        male_features = male_features,
        imp = importance_scores, 
        group_effect_all = average_treatment_effect(cf),
        group_effect_control = average_treatment_effect(cf, target.sample = "control"),
        group_effect_treated = average_treatment_effect(cf, target.sample = "treated"),
        group_effect_overlap = average_treatment_effect(cf, target.sample = "overlap"),
        evaluate_group_effect = test_calibration(cf),
        linear_projection_all = best_linear_projection(cf, X),
        linear_projection_overlap = best_linear_projection(cf, X, target.sample="overlap")
    )
    
    saveRDS(model_results, glue("{directory_path}/gender_{mental_health}_model_results.rds")) ## to check results at any time

    
 # plot partial dependence plots for top features 
    direct_path <- "../figures/PDP/gender"
    if (!dir.exists(direct_path)) {
        dir.create(direct_path, recursive = TRUE)
    }
    
    for (cov2plot in top_features) {
        # update feature name from the mapping
        # Prepare data
        X <- data.table(X)  
        df_x_mean <- X[, lapply(.SD, mean)]
        n <- nrow(d1)  
        cov_rge <- X[, fivenum(get(cov2plot))]
        pred_vals <- seq(cov_rge[1], cov_rge[5], length.out = n)
        
        df_x_mean <- df_x_mean[rep(1, n)]
        df_x_mean[, (cov2plot) := pred_vals]
        
        # Predict
        cf_predict <- predict(cf, newdata = df_x_mean, estimate.variance = TRUE)
        setDT(cf_predict)
        cf_predict$v <- df_x_mean[[cov2plot]]
        cf_predict[, se := sqrt(variance.estimates)]
        
        if (cov2plot %in% names(mapping_vector)) {
            cov2plot_name <- mapping_vector[cov2plot]
        } else {
            cov2plot_name <- cov2plot
        }
        
        p <- ggplot(cf_predict, aes(x = v, y = predictions)) +

            geom_line(colour = "#7A1B6D") +
            labs(x = cov2plot_name, y = "Mean group difference ") +
            theme_minimal() +
            theme(axis.line = element_line(color = "lightgrey"), 
                  axis.title.x  = element_text(size = 24),
                  axis.title.y  = element_text(size = 20),
                  axis.text = element_text(size = 16),
                  axis.ticks = element_line(color = "lightgrey"),
                  panel.background = element_rect(fill = "white", colour = "white"),
                  panel.grid.major = element_blank(), 
                  panel.grid.minor = element_blank())
        
        
        file_name <- glue("{direct_path}/gender_{mental_health}_{cov2plot_name}_PDP.jpg")
        ggsave(file_name, plot = p, width = 8, height = 4, dpi = 300)

    }
    
}


# Apply function for each gender and mental health outcome
gender_group <- c("female", "male")
mental_health_outcomes <- c("Positive_mental_health_outcome",
                            "Mental_problems_outcome",
                            "Externalizing_outcome", 
                            "Internalizing_outcome")

for (subgroup in gender_group) {
    for (mental_health in mental_health_outcomes) {
        grf_top_features(gender, mental_health)
    }
}

