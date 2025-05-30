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
