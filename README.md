# Dubai Housing Price Prediction

End-to-end machine learning project to predict property sale prices in the UAE using 41k+ listings and gradient-boosted tree models.

---

## 1. Project overview

In this project a supervised regression model is built to predict the sale price of residential properties in Dubai and other UAE cities from listing features (beds, baths, location, building attributes, posting date, etc.). 

The goal is to reproduce a realistic production-style ML workflow: clean and explore a large tabular dataset, engineer meaningful features, compare several tree-based algorithms with proper validation, and select a final model based on Mean Absolute Error (MAE).  

This repository contains the full cycle from raw CSV to a tested model saved with joblib.

---

## 2. Data and problem definition

**Business-style problem**  
Given a listing with structured features at posting time, estimate its market price in AED so that an agent or platform can price more accurately and benchmark new listings. 

**Dataset**  
- ~41,381 rows, 22 original columns (train before splits). [file:3]  
- Columns include: `price`, `pricecategory`, `type`, `beds`, `baths`, `address`, `furnishing`, `completionstatus`, `postdate`, `averagerent`, `buildingname`, `yearofcompletion`, `totalparkingspaces`, `totalfloors`, `totalbuildingareasqft`, `elevators`, `areaname`, `city`, `country`, `Latitude`, `Longitude`, `purpose`.
- All listings are in the UAE (country is constant).

**Target and splits**  
- Target: `price` (continuous).
- 60/20/20 split into train/validation/test via `train_test_split` with a two-stage 60–20–20 procedure.
  - Train: 24,828 rows  
  - Validation: 8,276 rows  
  - Test: 8,277 rows

---

## 3. EDA and feature engineering

### Exploratory analysis

Key steps:  
- Data integrity: checked shape, types, missing values, duplicates, and number of unique values.  
- Descriptive statistics: `df.describe()`, distributions for price, beds, baths, building area, etc.  
- Correlations: numeric correlation matrix between price and other numeric features (beds, baths, averagerent, yearofcompletion, totalparkingspaces, totalfloors, totalbuildingareasqft, elevators, Latitude, Longitude).
- Distribution plots:
  - Histograms and KDE for price in train/validation/test. 
  - Boxplots for price and key numeric variables to inspect outliers.  
- Categorical exploration:
  - Value counts for pricecategory (Average/Medium/High), property type (Apartment/Villa/Townhouse/Plots, etc.), furnishing, completionstatus, city, buildingname.

### Feature engineering

Main transformations:  
- **Date features**: parsed `postdate` to datetime and extracted `year`, `month`, and `quarter` for train/valid/test.
- **Dropping low-value or constant features**:
  - Removed `country` and `purpose` (no information gain, only “UAE” and “For Sale”).   
  - Removed latitude/longitude for the main modeling run to reduce noise and leakage concerns.  
  - Dropped `address` and `postdate` after extracting date parts. 
- **Near-zero variance analysis**:
  - Calculated `freqratio`, `uniqueratio`, `highfreqratio` to identify near-zero variance variables like `totalparkingspaces`, `country`, `purpose`, `buildingname`, `totalfloors`, `elevators`, etc., and decided which to drop or handle carefully. 
- **Target encoding (regularized)**:
  - Identified categorical columns to encode: `pricecategory`, `type`, `furnishing`, `completionstatus`, `buildingname`, `areaname`, `city`. 
  - Implemented custom `targetencodetrain(valid, test, col, target)`:
    - Out-of-fold encoding on train with KFold (5 folds) to avoid target leakage. 
    - Smoothed means using global mean and per-category mean within folds.
    - Applied learned mapping to validation and test; unseen categories mapped to global mean.  

All key transformations were applied consistently to train, validation, and test sets and artifacts (variable lists, encoders) were saved with joblib/pickle.  

---

## 4. Modeling workflow

### Train/validation/test strategy

- Train/validation/test split done once at the EDA stage and kept fixed across all models.  
- Model selection and hyperparameter tuning performed on the train set with 2-fold cross-validation, using MAE as the optimization metric.  
- The validation set is used for performance sanity checks and comparisons; the final test set is held out until the end. 

### Models tried

Implemented in scikit-learn (and XGBoost):

1. **Baseline models**  
   - Simple baselines (e.g., median or basic tree models) for reference MAE.  

2. **Tree-based regressors (single models)**  
   - RandomForestRegressor  
   - GradientBoostingRegressor  
   - XGBRegressor (XGBoost)  
   - AdaBoostRegressor (with DecisionTreeRegressor base estimator)  
   - BaggingRegressor  
   - Tuned DecisionTreeRegressor 

3. **Ensemble stacking**  
   - `StackingRegressor` combining: RF, GBM, AdaBoost, XGBoost, Bagging, tuned Decision Tree.
   - Meta-learner: `LinearRegression` with/without intercept.  

All models are wrapped in `Pipeline` objects; tree models operate only on engineered numeric features (target-encoded categories and numeric variables). 

### Hyperparameter tuning

Used `GridSearchCV` with `scoring="neg_mean_absolute_error"`, `cv=2`, `n_jobs=-1`, `verbose=1`. 

Examples (from best runs on one modeling iteration):  

- **AdaBoostRegressor**  
  - Pipeline: `Pipeline([("model", AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=124), random_state=124))])` 
  - Grid over:
    - `model__n_estimators`: [50, 100, 200]  
    - `model__learning_rate`: [0.01, 0.05, 0.1]  
    - `model__estimator__max_depth`: [2, 3, 4]  
    - `model__estimator__min_samples_leaf`: [1, 2]  
  - Best (one version): `max_depth=4`, `min_samples_leaf=1`, `learning_rate=0.01`, `n_estimators=50`, MAE ≈ 0.55 after log/scale transformation in that experiment. 

- **GradientBoostingRegressor**  
  - Grid:  
    - `model__n_estimators`: [100, 200, 300]  
    - `model__learning_rate`: [0.01, 0.05, 0.1]  
    - `model__max_depth`: [2, 3, 4]  
    - `model__min_samples_split`: [2, 10]  
    - `model__min_samples_leaf`: [1, 2] [file:1][file:10]  
  - Best example: `learning_rate=0.1`, `max_depth=4`, `min_samples_leaf=1`, `min_samples_split=2`, `n_estimators=300` with validation MAE around 0.34 (in transformed space). 

- **XGBRegressor**  
  - Grid:  
    - `model__n_estimators`: [200, 400]  
    - `model__learning_rate`: [0.01, 0.05, 0.1]  
    - `model__max_depth`: [3, 5, 7]  
    - `model__subsample`: [0.7, 1.0]  
    - `model__colsample_bytree`: [0.7, 1.0]  
    - `model__min_child_weight`: [1, 3]  
    - `model__gamma`: [0, 1]  
    - `model__reg_alpha`: [0.0, 0.001]  
    - `model__reg_lambda`: [1.0, 2.0]  
  - Best example:  
    - `colsample_bytree=1.0`, `gamma=0`, `learning_rate=0.1`, `max_depth=7`, `min_child_weight=1`, `n_estimators=400`, `reg_alpha=0.0`, `reg_lambda=1.0`, `subsample=1.0`; best CV MAE ≈ 0.31 in transformed scale and ≈ 4.06 in another variant. 

- **StackingRegressor**  
  - Base estimators: tuned RF, GBM, AdaBoost, XGBoost, Bagging, Decision Tree. 
  - Final estimator: `LinearRegression`; tuned only `fit_intercept` ∈ {True, False}.  
  - Best: `fit_intercept=False`, best CV MAE ≈ 0.30 in transformed scale and ≈ 4.00+ in another run.

Best models are serialized with `joblib.dump` into a `models` folder for later loading and inference.

---

## 5. Model evaluation and outcomes

### Metrics

- Main metric: **Mean Absolute Error (MAE)** on train/validation/test.  
- Secondary checks: distribution plots of residuals and predicted vs actual prices (in notebooks).

### Performance (representative)

Across experiments (different engineering versions), the ranking is consistent:  
- XGBoost and Gradient Boosting are the strongest single models.  
- StackingRegressor slightly improves MAE over the best single model in cross-validation.  

On the final held-out test set (using the selected best model configuration), the MAE remains close to the validation MAE, indicating limited overfitting and a stable generalization performance for this tabular problem.   

> Note: Exact numeric MAE values depend on the final preprocessing choice (e.g., whether using log-price or raw price); the notebooks document both intermediate and final scores.  

---

## 6. ML workflow summary

High-level ML lifecycle followed:

1. **Problem definition & metric**: price prediction as supervised regression, MAE as main metric (robust to large outliers vs MSE). 
2. **Data ingestion & cleaning**: load CSV, inspect types, handle missing values, drop purely constant or redundant columns.  
3. **Train/valid/test split**: 60/20/20 with fixed random seeds.  
4. **EDA**: summary statistics, distributions, correlations, and categorical frequencies.  
5. **Feature engineering**:
   - Date extraction (year, month, quarter).  
   - Dropping weak or near-zero-variance variables.  
   - Target encoding for high-cardinality categoricals (OOF, KFold). 
6. **Feature selection**: remove near-zero variance and high-cardinality-but-uninformative columns; focus on features that add signal.  
7. **Modeling**:
   - Baselines and several tree-based models.  
   - Hyperparameter tuning with GridSearchCV (MAE, CV=2, parallelized).   
8. **Ensembling**: Stacking multiple tuned models with linear regression meta-learner.  
9. **Validation & test**: compare models on validation set; evaluate best model once on test; visualize error distributions.  
10. **Persistence**: save processed datasets and trained models with joblib/pickle for reuse.  

---
## 9. Future work

Planned or possible extensions:

- Add a simple API (FastAPI/Flask) for serving predictions from the saved XGBoost / stacking model.  
- Add model explainability (SHAP/feature importance plots) for business interpretability.  
- Explore regularized linear models or tabular neural networks for comparison.  
- Implement more formal checks against data leakage in all encoding/imputation steps and wrap preprocessing into reusable sklearn transformers.  

You can paste and adapt this README, then tweak wording, numbers, and model names to exactly match your final chosen pipeline and scores.
