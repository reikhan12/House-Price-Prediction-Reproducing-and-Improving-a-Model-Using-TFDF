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

