
# House Price Prediction: Reproducing and Improving a Model in R

**Course**: Reproducible Research  
**Major**: Data Science and Business Analytics  
**Date**: 31.03.2025

**Team Members**  
- Maryam Abdulhuseynova (436856)  
- Parvin Badalov (456028)  
- Reikhan Gurbanova (468193)

---

## About This Project

This project started as part of our Reproducible Research course, where we were challenged to take an existing machine learning analysis and **reproduce it in a new programming environment**. We chose a [Kaggle Notebook](https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf) by *gusthema*, which predicts house prices using **TensorFlow Decision Forests (TFDF)** in Python.

However, we encountered a limitation — TensorFlow wasn’t available for use in our R setup. So instead, we pivoted to **reproducing and improving the model using classic ensemble techniques in R**, namely:

-  Random Forest  
-  XGBoost  
-  GBM (Gradient Boosting Machine)  
-  An Ensemble Model combining all three

---

##  Project Goals

1. **Translate** a machine learning pipeline from Python to R  
2. **Compare** the results across languages and methods (RMSE, R²)  
3. **Improve** the original approach with new algorithms and feature engineering  
4. **Test** the robustness of the model through cross-validation  
5. **Document** everything in a clean, reproducible workflow

---

##  Tools & Technologies

- **Language**: R (RStudio)  
- **Key Packages**: `randomForest`, `xgboost`, `gbm`, `tidyverse`, `caret`, `VIM`, `mice`, `corrplot`  
- **Version Control**: Git & GitHub

---

##  Dataset

We used the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), which includes 79 features describing the properties of homes in Ames, Iowa. It’s a great dataset for structured regression problems.

---

##  Workflow Highlights

### 1. Data Preprocessing
- Handled missing values using median, mode, and placeholder categories
- Cleaned and standardized both training and test datasets

### 2. Feature Engineering
- Created meaningful new variables: `TotalSF`, `TotalBath`, `GarageAge`, `HasPool`, etc.  
- Performed log transformation on skewed variables  
- Encoded ordinal quality variables (e.g., Excellent = 5, Poor = 1)

### 3. Model Building & Tuning
- Hyperparameter tuning using manual grid search  
- Evaluated models using RMSE, MAE, R², and MAPE  
- Built an ensemble using inverse RMSE-weighted averaging

### 4. Evaluation & Visualization
- Feature importance for each model  
- Residual diagnostics (QQ plots, histograms, residual vs fitted)  
- Correlation heatmaps and RMSE progress plots

---

##  Model Performance Summary



| Model           | RMSE (Validation) |
|------------------|------------------|
| Random Forest    | 22,486.44        |
| XGBoost          | 22,013.17        |
| GBM              | 21,900.43        |
| **Ensemble**     | **20,761.50**    |

> Ensemble model achieved the best performance by combining the strengths of all individual models.


> Exact values available in plots and script output

---

##  Repository Structure

```
├── data/                   <- Raw and processed data files
├── scripts/                <- All preprocessing and model code
├── outputs/                <- Generated plots, models, and predictions
├── Proposal RR.pdf         <- Original project proposal document
└── README.md               <- You're here!
```

---

##  How to Run It

1. Install required packages:
```r
install.packages(c("tidyverse", "randomForest", "xgboost", "gbm", "caret", 
                   "VIM", "mice", "corrplot", "gridExtra", 
                   "doParallel", "foreach", "pROC"))
```

2. Run the full pipeline:
```r
source("scripts/main_model_pipeline.R")
```

3. Outputs include:
- Visual comparisons
- Model diagnostics
- CSV prediction files
- Feature importances

---

##  Why We Chose This Project

We wanted something practical and well-scoped — a problem where we could explore **reproducibility, collaboration, and critical evaluation**. This project let us:

- Practice **code translation** across languages  
- Examine **reproducibility and robustness**  
- Improve model accuracy with **custom enhancements**  
- Collaborate using **Git** and document everything end-to-end

---

##  Final Deliverables

-  Reproduced R code based on a Python TFDF notebook  
-  Improved models with hyperparameter tuning and engineered features  
-  Visual model comparisons and diagnostics  
-  GitHub repo with version control, documentation, and reproducible scripts  
-  Final reflection in class + short write-up of challenges and learnings

---

## 🌳 Git Commit Tree

Below is the visual Git commit tree of the `main` branch, including merges from feature branches:

* 80cf436 (HEAD -> main, origin/main, origin/HEAD) Merge pull request #15 from reikhan12/maryam-reproducibility
|\
| * 9d60ccd Update README.md
* | 19e4225 Merge pull request #14 from reikhan12/Reikhan
|\ \
| * | 38b771c Add files via upload
* | | 38e995e Merge pull request #13 from reikhan12/Reikhan
|\| |
| * | 2df9779 house_pricing.R
| * | 35e9db6 house_pricing.R
* | | 7d02672 Merge pull request #12 from reikhan12/Reikhan
|\| |
| * | 193fc63 Add files via upload
|/ /
* | 4a35fe5 (origin/Reikhan) Removed incorrect submission files from outputs folder
* | 642f4c0 (Reikhan) Add files via upload
* | 0b26f3f Merge pull request #11 from reikhan12/Reikhan
|\ \
| * | 722bd70 house_pricing.R
|/ /
* | dbd8fb6 Merge pull request #10 from BadalovP/Development_Parvin.B
|\
| * 7b017d9 Added heatmap
* | 7d3e31e Merge pull request #9 from BadalovP/Development_Parvin.B
|\|
| * 551afd4 Final updates to development branch before merging to main
| * f60d7c5 Merge remote-tracking branch 'upstream/main'
* 8a62575 Merge pull request #8 from reikhan12/Reikhan
|\
| * b19d5b8 house_pricing.R
* 93a7ec4 Merge pull request #5 from BadalovP/Development_Parvin.B

