# House Prices Prediction using Multiple Models in R
# Enhanced version with Random Forest, XGBoost, Gradient Boosting, Hybrid Model
# Feature Engineering and Hyperparameter Tuning
install.packages(c("tidyverse", "randomForest", "xgboost", "gbm", "caret", "VIM", "mice", "corrplot", "gridExtra", "doParallel", "foreach", "pROC"), dependencies = TRUE)
install.packages("renv")
# Load required libraries
library(tidyverse)
library(randomForest)
library(xgboost)
library(gbm)
library(caret)
library(VIM)
library(mice)
library(corrplot)
library(gridExtra)
library(pROC)
library(doParallel)
library(foreach)
library(renv)

# Set up parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Set seed for reproducibility
set.seed(42)

# Load the dataset
#setwd("C:/Users/User/Desktop/R_R/House-Price-Prediction-Reproducing-and-Improving-a-Model-Using-TFDF")

train_data <- read.csv("data/train.csv", stringsAsFactors = FALSE)
test_data <- read.csv("data/test.csv", stringsAsFactors = FALSE)

cat("Full train dataset shape:", dim(train_data), "\n")
cat("Test dataset shape:", dim(test_data), "\n")

# Remove Id column from training data (keep for test data)
train_data <- train_data %>% select(-Id)
test_ids <- test_data$Id
test_data <- test_data %>% select(-Id)

# Basic data exploration
cat("\nTarget variable (SalePrice) summary:\n")
summary(train_data$SalePrice)

# Visualize target distribution (equivalent to Python's sns.distplot)
p1 <- ggplot(train_data, aes(x = SalePrice)) +
  geom_histogram(aes(y = ..density..), bins = 100, fill = "green", alpha = 0.4) +
  geom_density(color = "darkgreen", size = 1) +
  theme_minimal() +
  ggtitle("Distribution of Sale Prices") +
  xlab("SalePrice") +
  ylab("Density")

p2 <- ggplot(train_data, aes(x = log(SalePrice))) +
  geom_histogram(aes(y = ..density..), bins = 50, fill = "lightgreen", alpha = 0.7) +
  geom_density(color = "darkgreen", size = 1) +
  theme_minimal() +
  ggtitle("Distribution of Log Sale Prices") +
  xlab("Log(SalePrice)") +
  ylab("Density")

grid.arrange(p1, p2, ncol = 2)

# Display data types (equivalent to Python's list(set(dataset_df.dtypes.tolist())))
cat("\nData types in the dataset:\n")
data_types <- sapply(train_data, class)
unique_types <- unique(data_types)
cat("Unique data types:", paste(unique_types, collapse = ", "), "\n")

# Select numerical columns and create histograms (equivalent to Python's df_num.hist())
cat("\nCreating histograms for numerical variables...\n")
numerical_cols <- train_data %>% 
  select_if(is.numeric) %>%
  select(-SalePrice)  # Remove target variable for this visualization

# Create a more manageable subset of numerical variables for plotting
num_cols_subset <- numerical_cols %>%
  select(LotArea, OverallQual, OverallCond, YearBuilt, YearRemodAdd, 
         X1stFlrSF, X2ndFlrSF, LowQualFinSF, GrLivArea, FullBath, 
         HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd, 
         Fireplaces, GarageYrBlt, GarageCars, GarageArea, WoodDeckSF, 
         OpenPorchSF, EnclosedPorch, X3SsnPorch, ScreenPorch, PoolArea)

# Convert to long format for ggplot
num_long <- num_cols_subset %>%
  gather(key = "Variable", value = "Value")

# Create histogram plot
p_hist <- ggplot(num_long, aes(x = Value)) +
  geom_histogram(bins = 50, fill = "lightblue", alpha = 0.7) +
  facet_wrap(~Variable, scales = "free", ncol = 6) +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 8),
        axis.text.y = element_text(size = 8),
        strip.text = element_text(size = 8)) +
  ggtitle("Histograms of Numerical Variables")

print(p_hist)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

feature_engineering <- function(data, is_train = TRUE) {
  
  # 1. Handle missing values
  cat("Handling missing values...\n")
  
  # Categorical variables - fill with 'None' or mode
  cat_vars_none <- c("Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 
                     "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",  # nolint # nolint: line_length_linter.
                     "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature") # nolint: line_length_linter.
  
  for (var in cat_vars_none) {
    if (var %in% names(data)) {
      data[[var]][is.na(data[[var]])] <- "None"
    }
  }
  
  # Fill other categorical variables with mode
  cat_vars <- sapply(data, function(x) is.character(x) | is.factor(x))
  for (var in names(data)[cat_vars]) {
    if (sum(is.na(data[[var]])) > 0) {
      mode_val <- names(sort(table(data[[var]]), decreasing = TRUE))[1]
      data[[var]][is.na(data[[var]])] <- mode_val
    }
  }
  
  # Numerical variables - fill with median or 0 for specific cases
  num_vars_zero <- c("LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", 
                     "BsmtUnfSF", "TotalBsmtSF", "GarageYrBlt", "GarageArea", "GarageCars")
  
  for (var in num_vars_zero) {
    if (var %in% names(data)) {
      data[[var]][is.na(data[[var]])] <- 0
    }
  }
  
  # Fill remaining numerical variables with median
  num_vars <- sapply(data, is.numeric)
  for (var in names(data)[num_vars]) {
    if (sum(is.na(data[[var]])) > 0) {
      data[[var]][is.na(data[[var]])] <- median(data[[var]], na.rm = TRUE)
    }
  }
  
  # 2. Create new features
  cat("Creating new features...\n")
  
  # Total square footage
  data$TotalSF <- data$X1stFlrSF + data$X2ndFlrSF + data$TotalBsmtSF
  
  # Total bathrooms
  data$TotalBath <- data$FullBath + 0.5 * data$HalfBath + data$BsmtFullBath + 0.5 * data$BsmtHalfBath
  
  # House age
  data$HouseAge <- data$YrSold - data$YearBuilt
  data$YearsSinceRemodel <- data$YrSold - data$YearRemodAdd
  
  # Garage age
  data$GarageAge <- ifelse(data$GarageYrBlt == 0, 0, data$YrSold - data$GarageYrBlt)
  
  # Porch area
  data$PorchSF <- data$OpenPorchSF + data$EnclosedPorch + data$X3SsnPorch + data$ScreenPorch
  
  # Quality scores
  data$OverallScore <- data$OverallQual * data$OverallCond
  
  # Basement indicator
  data$HasBasement <- ifelse(data$TotalBsmtSF > 0, 1, 0)
  
  # Garage indicator
  data$HasGarage <- ifelse(data$GarageArea > 0, 1, 0)
  
  # Pool indicator
  data$HasPool <- ifelse(data$PoolArea > 0, 1, 0)
  
  # Fireplace indicator
  data$HasFireplace <- ifelse(data$Fireplaces > 0, 1, 0)
  
  # 3. Feature transformations
  cat("Applying feature transformations...\n")
  
  # Log transform skewed numerical features
  skewed_features <- c("LotArea", "X1stFlrSF", "GrLivArea", "TotalSF")
  for (feature in skewed_features) {
    if (feature %in% names(data)) {
      data[[paste0(feature, "_log")]] <- log1p(data[[feature]])
    }
  }
  
  # 4. Encode categorical variables
  cat("Encoding categorical variables...\n")
  
  # Convert character variables to factors
  char_vars <- sapply(data, is.character)
  data[char_vars] <- lapply(data[char_vars], as.factor)
  
  # Ordinal encoding for quality variables
  qual_map <- c("None" = 0, "Po" = 1, "Fa" = 2, "TA" = 3, "Gd" = 4, "Ex" = 5)
  
  qual_vars <- c("ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", 
                 "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC")
  
  for (var in qual_vars) {
    if (var %in% names(data)) {
      data[[var]] <- as.numeric(qual_map[as.character(data[[var]])])
      data[[var]][is.na(data[[var]])] <- 0
    }
  }
  
  return(data)
}

# Apply feature engineering
cat("Applying feature engineering to training data...\n")
train_processed <- feature_engineering(train_data, is_train = TRUE)

cat("Applying feature engineering to test data...\n")
test_processed <- feature_engineering(test_data, is_train = FALSE)

# Ensure both datasets have same columns (except target)
common_features <- intersect(names(train_processed), names(test_processed))
train_processed <- train_processed[, c(common_features, "SalePrice")]
test_processed <- test_processed[, common_features]

# =============================================================================
# PREPARE DATA FOR MODELING
# =============================================================================

# Split training data
set.seed(42)
train_index <- createDataPartition(train_processed$SalePrice, p = 0.7, list = FALSE)
train_set <- train_processed[train_index, ]
valid_set <- train_processed[-train_index, ]

# Prepare features and target
train_features <- train_set %>% select(-SalePrice)
train_target <- train_set$SalePrice

valid_features <- valid_set %>% select(-SalePrice)
valid_target <- valid_set$SalePrice

# Convert categorical variables to dummy variables for some models
train_matrix <- model.matrix(SalePrice ~ ., data = train_set)[, -1]
valid_matrix <- model.matrix(SalePrice ~ ., data = valid_set)[, -1]

# Prepare test data
# Prepare test data - fix for factor levels issue
# First, ensure all factor variables in test data have same levels as training data
factor_cols <- sapply(train_processed, is.factor)
factor_names <- names(train_processed)[factor_cols]

for (col in factor_names) {
  if (col %in% names(test_processed) && col != "SalePrice") {
    # Get levels from training data
    train_levels <- levels(train_processed[[col]])
    # Set same levels for test data
    test_processed[[col]] <- factor(test_processed[[col]], levels = train_levels)
  }
}

# Remove any columns that have only one level (these cause the contrasts error)
test_processed_clean <- test_processed[, sapply(test_processed, function(x) {
  if(is.factor(x)) {
    return(nlevels(x) > 1)
  } else {
    return(TRUE)
  }
})]

# Now create the model matrix
test_matrix <- model.matrix(~ ., data = test_processed_clean)[, -1]

# Ensure test matrix has same columns as training matrix
missing_cols <- setdiff(colnames(train_matrix), colnames(test_matrix))
for (col in missing_cols) {
  test_matrix <- cbind(test_matrix, 0)
  colnames(test_matrix)[ncol(test_matrix)] <- col
}

# Remove extra columns and reorder
common_cols <- intersect(colnames(test_matrix), colnames(train_matrix))
test_matrix <- test_matrix[, common_cols]

# Ensure same column order as training matrix
test_matrix <- test_matrix[, colnames(train_matrix)[colnames(train_matrix) %in% colnames(test_matrix)]]

# =============================================================================
# MODEL 1: RANDOM FOREST WITH HYPERPARAMETER TUNING
# =============================================================================

cat("\n=== Training Random Forest Model ===\n")

# Hyperparameter tuning for Random Forest
rf_grid <- expand.grid(
  mtry = c(5, 10, 15, 20),
  ntree = c(100, 300, 500),
  nodesize = c(5, 10, 15)
)

best_rf_rmse <- Inf
best_rf_params <- NULL
best_rf_model <- NULL

for (i in 1:nrow(rf_grid)) {
  cat(sprintf("RF Grid Search: %d/%d\n", i, nrow(rf_grid)))
  
  rf_model <- randomForest(
    x = train_features,
    y = train_target,
    mtry = rf_grid$mtry[i],
    ntree = rf_grid$ntree[i],
    nodesize = rf_grid$nodesize[i],
    importance = TRUE
  )
  
  rf_pred <- predict(rf_model, valid_features)
  rf_rmse <- sqrt(mean((valid_target - rf_pred)^2))
  
  if (rf_rmse < best_rf_rmse) {
    best_rf_rmse <- rf_rmse
    best_rf_params <- rf_grid[i, ]
    best_rf_model <- rf_model
  }
}

cat("Best RF RMSE:", best_rf_rmse, "\n")
cat("Best RF Parameters:\n")
print(best_rf_params)

# =============================================================================
# MODEL 2: XGBOOST WITH HYPERPARAMETER TUNING
# =============================================================================

cat("\n=== Training XGBoost Model ===\n")

# Prepare DMatrix for XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_target)
dvalid <- xgb.DMatrix(data = valid_matrix, label = valid_target)
dtest <- xgb.DMatrix(data = test_matrix)

# Hyperparameter tuning for XGBoost
xgb_grid <- expand.grid(
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(3, 6, 9),
  subsample = c(0.8, 0.9),
  colsample_bytree = c(0.8, 0.9)
)

best_xgb_rmse <- Inf
best_xgb_params <- NULL
best_xgb_model <- NULL

for (i in 1:nrow(xgb_grid)) {
  cat(sprintf("XGBoost Grid Search: %d/%d\n", i, nrow(xgb_grid)))
  
  xgb_params <- list(
    objective = "reg:squarederror",
    eta = xgb_grid$eta[i],
    max_depth = xgb_grid$max_depth[i],
    subsample = xgb_grid$subsample[i],
    colsample_bytree = xgb_grid$colsample_bytree[i]
  )
  
  xgb_model <- xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 100,
    watchlist = list(train = dtrain, valid = dvalid),
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  xgb_pred <- predict(xgb_model, dvalid)
  xgb_rmse <- sqrt(mean((valid_target - xgb_pred)^2))
  
  if (xgb_rmse < best_xgb_rmse) {
    best_xgb_rmse <- xgb_rmse
    best_xgb_params <- xgb_grid[i, ]
    best_xgb_model <- xgb_model
  }
}

cat("Best XGBoost RMSE:", best_xgb_rmse, "\n")
cat("Best XGBoost Parameters:\n")
print(best_xgb_params)

# =============================================================================
# MODEL 3: GRADIENT BOOSTING MACHINE (GBM)
# =============================================================================

cat("\n=== Training GBM Model ===\n")

# Hyperparameter tuning for GBM
gbm_grid <- expand.grid(
  n.trees = c(100, 300, 500),
  interaction.depth = c(3, 6, 9),
  shrinkage = c(0.01, 0.05, 0.1),
  n.minobsinnode = c(10, 20)
)

best_gbm_rmse <- Inf
best_gbm_params <- NULL
best_gbm_model <- NULL

# Sample a subset of grid for faster computation
set.seed(42)
gbm_sample <- sample(nrow(gbm_grid), min(15, nrow(gbm_grid)))

for (i in gbm_sample) {
  cat(sprintf("GBM Grid Search: %d/%d\n", which(gbm_sample == i), length(gbm_sample)))
  
  gbm_model <- gbm(
    SalePrice ~ .,
    data = train_set,
    distribution = "gaussian",
    n.trees = gbm_grid$n.trees[i],
    interaction.depth = gbm_grid$interaction.depth[i],
    shrinkage = gbm_grid$shrinkage[i],
    n.minobsinnode = gbm_grid$n.minobsinnode[i],
    cv.folds = 5,
    verbose = FALSE
  )
  
  best_iter <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
  gbm_pred <- predict(gbm_model, valid_set, n.trees = best_iter)
  gbm_rmse <- sqrt(mean((valid_target - gbm_pred)^2))
  
  if (gbm_rmse < best_gbm_rmse) {
    best_gbm_rmse <- gbm_rmse
    best_gbm_params <- gbm_grid[i, ]
    best_gbm_model <- gbm_model
  }
}

cat("Best GBM RMSE:", best_gbm_rmse, "\n")
cat("Best GBM Parameters:\n")
print(best_gbm_params)

# =============================================================================
# MODEL 4: HYBRID/ENSEMBLE MODEL
# =============================================================================

cat("\n=== Creating Hybrid/Ensemble Model ===\n")

# Get predictions from all models on validation set
rf_valid_pred <- predict(best_rf_model, valid_features)
xgb_valid_pred <- predict(best_xgb_model, dvalid)
gbm_valid_pred <- predict(best_gbm_model, valid_set, n.trees = gbm.perf(best_gbm_model, method = "cv", plot.it = FALSE))

# Create ensemble predictions with weighted average
# Weights based on individual model performance (inverse of RMSE)
rf_weight <- 1 / best_rf_rmse
xgb_weight <- 1 / best_xgb_rmse
gbm_weight <- 1 / best_gbm_rmse

total_weight <- rf_weight + xgb_weight + gbm_weight

rf_weight_norm <- rf_weight / total_weight
xgb_weight_norm <- xgb_weight / total_weight
gbm_weight_norm <- gbm_weight / total_weight

ensemble_pred <- (rf_weight_norm * rf_valid_pred + 
                  xgb_weight_norm * xgb_valid_pred + 
                  gbm_weight_norm * gbm_valid_pred)

ensemble_rmse <- sqrt(mean((valid_target - ensemble_pred)^2))

cat("Ensemble RMSE:", ensemble_rmse, "\n")
cat("Ensemble Weights - RF:", round(rf_weight_norm, 3), 
    "XGB:", round(xgb_weight_norm, 3), 
    "GBM:", round(gbm_weight_norm, 3), "\n")

# =============================================================================
# MODEL COMPARISON
# =============================================================================

cat("\n=== Model Performance Comparison ===\n")
model_results <- data.frame(
  Model = c("Random Forest", "XGBoost", "GBM", "Ensemble"),
  RMSE = c(best_rf_rmse, best_xgb_rmse, best_gbm_rmse, ensemble_rmse),
  stringsAsFactors = FALSE
)

model_results <- model_results[order(model_results$RMSE), ]
print(model_results)

# =============================================================================
# VISUALIZATION OF TRAINING PROGRESS AND MODEL INSIGHTS
# =============================================================================

cat("\n=== Creating Model Visualization and Training Insights ===\n")

# 1. Random Forest: Out-of-bag error progression (equivalent to Python's training logs plot)
cat("Plotting Random Forest OOB Error progression...\n")

# Create a function to extract OOB error progression
extract_rf_oob <- function(rf_model) {
  oob_error <- rf_model$mse
  trees <- 1:length(oob_error)
  rmse <- sqrt(oob_error)
  return(data.frame(Trees = trees, RMSE = rmse))
}

rf_progress <- extract_rf_oob(best_rf_model)
p_rf_progress <- ggplot(rf_progress, aes(x = Trees, y = RMSE)) +
  geom_line(color = "blue", size = 1) +
  theme_minimal() +
  ggtitle("Random Forest: RMSE vs Number of Trees (OOB)") +
  xlab("Number of Trees") +
  ylab("RMSE (Out-of-bag)")

print(p_rf_progress)

# 2. XGBoost: Training progress plot
cat("Plotting XGBoost training progress...\n")

# Extract training history from XGBoost model
xgb_eval_log <- best_xgb_model$evaluation_log
if (!is.null(xgb_eval_log)) {
  p_xgb_progress <- ggplot(xgb_eval_log, aes(x = iter)) +
    geom_line(aes(y = train_rmse, color = "Training"), size = 1) +
    geom_line(aes(y = valid_rmse, color = "Validation"), size = 1) +
    scale_color_manual(values = c("Training" = "blue", "Validation" = "red")) +
    theme_minimal() +
    ggtitle("XGBoost: Training vs Validation RMSE") +
    xlab("Iteration") +
    ylab("RMSE") +
    labs(color = "Dataset")
  
  print(p_xgb_progress)
}

# 3. Model Tree Visualization (equivalent to Python's plot_model_in_colab)
cat("Creating Random Forest tree visualization...\n")

# Extract first tree structure (simplified representation)
# Note: R doesn't have direct equivalent to Python's model plotter, so we'll create feature importance plot instead
rf_tree_importance <- data.frame(
  Feature = names(best_rf_model$importance[,1]),
  Importance = best_rf_model$importance[,1]
) %>%
  arrange(desc(Importance)) %>%
  head(15)

p_tree_structure <- ggplot(rf_tree_importance, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "forestgreen", alpha = 0.7) +
  coord_flip() +
  theme_minimal() +
  ggtitle("Random Forest: Feature Importance (Tree Structure Insight)") +
  xlab("Features") +
  ylab("Mean Decrease in MSE")

print(p_tree_structure)

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS (Enhanced with more visualizations)
# =============================================================================

cat("\n=== Feature Importance Analysis ===\n")

# Random Forest Feature Importance (equivalent to Python's NUM_AS_ROOT visualization)
rf_importance <- importance(best_rf_model)
rf_imp_df <- data.frame(
  Feature = rownames(rf_importance),
  IncMSE = rf_importance[, "%IncMSE"],
  IncNodePurity = rf_importance[, "IncNodePurity"],
  stringsAsFactors = FALSE
) %>%
  arrange(desc(IncMSE)) %>%
  head(20)

# Create horizontal bar plot (equivalent to Python's barh plot)
p_rf_importance <- ggplot(rf_imp_df, aes(x = reorder(Feature, IncMSE), y = IncMSE)) +
  geom_col(fill = "skyblue", alpha = 0.8) +
  coord_flip() +
  theme_minimal() +
  ggtitle("Random Forest: Feature Importance (%IncMSE)") +
  xlab("Features") +
  ylab("%IncMSE") +
  theme(axis.text.y = element_text(size = 9))

# Add value labels on bars (equivalent to Python's text annotations)
p_rf_importance <- p_rf_importance +
  geom_text(aes(label = sprintf("%.4f", IncMSE)), 
            hjust = -0.1, size = 3)

print(p_rf_importance)

# XGBoost Feature Importance with gain values
xgb_importance <- xgb.importance(model = best_xgb_model)
xgb_imp_df <- xgb_importance %>%
  head(20) %>%
  select(Feature, Gain) %>%
  rename(Importance = Gain)

p_xgb_importance <- ggplot(xgb_imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "lightgreen", alpha = 0.8) +
  coord_flip() +
  theme_minimal() +
  ggtitle("XGBoost: Feature Importance (Gain)") +
  xlab("Features") +
  ylab("Gain") +
  theme(axis.text.y = element_text(size = 9)) +
  geom_text(aes(label = sprintf("%.4f", Importance)), 
            hjust = -0.1, size = 3)

print(p_xgb_importance)

# Combined feature importance comparison
grid.arrange(p_rf_importance, p_xgb_importance, ncol = 2)

# 4. Correlation heatmap of top features (additional insight)
cat("Creating correlation heatmap of top features...\n")

top_features <- c("GrLivArea", "TotalBsmtSF", "X1stFlrSF", "FullBath", 
                  "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd", "GarageArea", 
                  "GarageCars", "OverallQual", "SalePrice")

# Select available features
available_features <- intersect(top_features, names(train_data))
cor_data <- train_data[, available_features]

# Calculate correlation matrix
cor_matrix <- cor(cor_data, use = "complete.obs")

# Create correlation plot
library(reshape2)
cor_melted <- melt(cor_matrix)

p_corr <- ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  ggtitle("Correlation Heatmap of Key Features") +
  xlab("") + ylab("")

print(p_corr)

# =============================================================================
# GENERATE PREDICTIONS ON TEST SET
# =============================================================================

cat("\n=== Generating Test Predictions ===\n")

# Get predictions from all models on test set
rf_test_pred <- predict(best_rf_model, test_processed)
xgb_test_pred <- predict(best_xgb_model, dtest)
gbm_test_pred <- predict(best_gbm_model, test_processed, n.trees = gbm.perf(best_gbm_model, method = "cv", plot.it = FALSE))

# Create ensemble predictions
ensemble_test_pred <- (rf_weight_norm * rf_test_pred + 
                       xgb_weight_norm * xgb_test_pred + 
                       gbm_weight_norm * gbm_test_pred)

# Create submission files for each model
submissions <- list(
  "rf_submission.csv" = data.frame(Id = test_ids, SalePrice = rf_test_pred),
  "xgb_submission.csv" = data.frame(Id = test_ids, SalePrice = xgb_test_pred),
  "gbm_submission.csv" = data.frame(Id = test_ids, SalePrice = gbm_test_pred),
  "ensemble_submission.csv" = data.frame(Id = test_ids, SalePrice = ensemble_test_pred)
)

# Write submission files
for (filename in names(submissions)) {
  write.csv(submissions[[filename]], filename, row.names = FALSE)
  cat("Saved:", filename, "\n")
}

# =============================================================================
# RESIDUAL ANALYSIS AND MODEL DIAGNOSTICS
# =============================================================================

cat("\n=== Residual Analysis and Model Diagnostics ===\n")

# Calculate residuals for ensemble model
residuals <- valid_target - ensemble_pred

# 1. Residuals vs Fitted Values (equivalent to Python's scatter plot)
p_resid_fitted <- ggplot(data.frame(Fitted = ensemble_pred, Residuals = residuals), 
             aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", size = 1) +
  geom_smooth(se = FALSE, color = "green") +
  theme_minimal() +
  ggtitle("Ensemble Model: Residuals vs Fitted Values") +
  xlab("Fitted Values") +
  ylab("Residuals")

print(p_resid_fitted)

# 2. Distribution of Residuals (equivalent to Python's histogram)
p_resid_hist <- ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  theme_minimal() +
  ggtitle("Distribution of Residuals") +
  xlab("Residuals") +
  ylab("Density")

print(p_resid_hist)

# 3. Q-Q plot for normality check
p_qq <- ggplot(data.frame(Residuals = residuals), aes(sample = Residuals)) +
  stat_qq() +
  stat_qq_line(color = "red") +
  theme_minimal() +
  ggtitle("Q-Q Plot of Residuals") +
  xlab("Theoretical Quantiles") +
  ylab("Sample Quantiles")

print(p_qq)

# 4. Actual vs Predicted scatter plot
p_actual_pred <- ggplot(data.frame(Actual = valid_target, Predicted = ensemble_pred), 
                        aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
  theme_minimal() +
  ggtitle("Actual vs Predicted Values") +
  xlab("Actual SalePrice") +
  ylab("Predicted SalePrice")

print(p_actual_pred)

# Create a 2x2 grid of diagnostic plots
grid.arrange(p_resid_fitted, p_resid_hist, p_qq, p_actual_pred, ncol = 2)

# Calculate additional metrics
mae <- mean(abs(residuals))
r_squared <- 1 - sum(residuals^2) / sum((valid_target - mean(valid_target))^2)
mape <- mean(abs(residuals/valid_target)) * 100

cat("Additional Metrics for Ensemble Model:\n")
cat("MAE:", round(mae, 2), "\n")
cat("R-squared:", round(r_squared, 4), "\n")
cat("MAPE:", round(mape, 2), "%\n")

# 5. Model Performance Summary Plot
model_performance <- data.frame(
  Model = c("Random Forest", "XGBoost", "GBM", "Ensemble"),
  RMSE = c(best_rf_rmse, best_xgb_rmse, best_gbm_rmse, ensemble_rmse),
  stringsAsFactors = FALSE
)

p_model_comp <- ggplot(model_performance, aes(x = reorder(Model, -RMSE), y = RMSE)) +
  geom_col(fill = c("skyblue", "lightgreen", "orange", "purple"), alpha = 0.8) +
  geom_text(aes(label = round(RMSE, 0)), vjust = -0.5, size = 4) +
  theme_minimal() +
  ggtitle("Model Performance Comparison (RMSE)") +
  xlab("Models") +
  ylab("RMSE") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p_model_comp)

# Stop parallel processing
stopCluster(cl)

cat("\n=== Analysis Complete ===\n")
cat("Best performing model:", model_results$Model[1], "\n")
cat("Best RMSE:", round(model_results$RMSE[1], 2), "\n")

