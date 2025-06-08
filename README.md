# 🧬 Aging Biomarkers – Methylation-Based Chronological Age Estimator  
**Leveraging DNA methylation data and machine learning to predict human age**

---

## 🌟 Project Summary

This project develops machine learning models to estimate **chronological age** from **DNA methylation profiles** using high-dimensional CpG site data. It leverages two public datasets from NCBI GEO and employs GPU-accelerated feature selection, stratified sampling, and regression modeling to build accurate and generalizable age predictors.

> 🔍 _Why does this matter?_  
> Accurate biological age estimation has actionable implications in:
> - 🧬 **Healthcare**: Early identification of accelerated aging and disease risk  
> - 🕵️ **Forensics**: Age profiling from biological samples  
> - 🧪 **Research**: Studying epigenetic effects of lifestyle and interventions  

---

## 💡 Here's Why I’m Doing This

Chronological age doesn’t always reflect biological aging. Individuals age differently due to lifestyle, genetics, and environmental factors. This project aims to bridge that gap by modeling **DNA methylation**—a robust molecular marker of biological aging.

Using CpG methylation patterns, the models can:
- Identify early signs of aging-related decline
- Quantify effects of lifestyle and intervention
- Assist in sample profiling when metadata is missing

This project is a step toward applying **genomics + machine learning** for interpretable, personalized, and scalable insights into how we age.

---

## 📦 Datasets Used

- **GSE40279**: Primary training and test set  
- **GSE157131**: External evaluation set  

_Sourced from NCBI GEO. Please refer to original dataset licenses and citations._

---

## ❓ Why It Matters

- 🔬 **Scientific Insight**: Maps molecular aging markers in blood  
- 🏥 **Clinical Relevance**: Predicts healthspan, informs risk  
- 📈 **Societal Impact**: Supports forensic and demographic applications  
- 💼 **Industry Relevance**: Fuels innovation in longevity tech, health analytics  

---

## 💥 The Challenge

Epigenetic age prediction must overcome:
- ⚖️ **High dimensionality** of CpG features (400k+)  
- 🌍 **Generalization** across populations and age ranges  
- 🧠 **Interpretability** in clinical/forensic applications  

This project introduces a scalable, GPU-accelerated ML pipeline that addresses each of these challenges through data-driven, biologically informed modeling.

---

## 🛠️ Tools & Technologies

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/-Seaborn-66B3BA?logo=python)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter)
![XGBoost](https://img.shields.io/badge/-XGBoost-EC5E24?logo=xgboost)
![CuPy](https://img.shields.io/badge/-CuPy-00BFFF?logo=nvidia)
![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github)

---

## 🧠 Scientific Terms & Definitions

| Term | Description |
|------|-------------|
| **Epigenetics** | Heritable changes in gene expression without altering DNA sequence. |
| **DNA Methylation** | Chemical tagging (–CH₃) on DNA that regulates gene activity. |
| **CpG Sites** | Regions where cytosine is followed by guanine; key methylation sites. |
| **Chronological Age** | Actual age in years. |
| **Biological Age** | Estimated age based on molecular/physiological biomarkers. |
| **MAE** | Mean Absolute Error — average absolute prediction error in years. |
| **MSE** | Mean Squared Error — penalizes larger errors, used as training objective. |

---

## 🧠 Key Techniques Used

| Step | Technique |
|------|-----------|
| **Feature Selection** | Bootstrap linear regression, LassoCV, CuPy-accelerated correlation filtering |
| **Feature Sets** | Compact (350), Balanced (800), Comprehensive (1,200 CpGs) |
| **Preprocessing** | RobustScaler + QuantileTransformer (normalized features) |
| **Sampling** | Stratified train-test split by age percentiles |
| **Optimization** | CuPy for fast computation and memory-efficient batch processing |
| **Model Evaluation** | MSE, MAE, R², visual diagnostics |
| **Deployment Ready** | Models and pipeline artifacts saved with `joblib` |

---

## 🔬 Methodology Overview

- Started with 473,034 CpGs from GSE40279  
- Selected top 10,000 by variance  
- Bootstrapped linear regression to stabilize coefficients  
- LassoCV reduced to ~405 CpGs  
- CuPy-based correlation ranking finalized top 1,200 CpGs  
- Preprocessing with scaling + quantile transform  
- Age-stratified train-test split (80/20)  
- Cross-validated Ridge and ElasticNet models trained using GridSearchCV  

---

## 🧪 Model Selection Criteria

| Model | Justification |
|-------|----------------|
| **RidgeCV** | Best generalization, resistant to multicollinearity |
| **ElasticNetCV** | Feature sparsity + L2 robustness in one model |
| **GridSearchCV** | Tuned `alpha` and `l1_ratio` with exhaustive validation |
| **Ensemble-Ready** | Modular feature pipeline for stacking & boosting models |

---

## 📊 Performance Metrics

| Metric | Description |
|--------|-------------|
| **MSE** | Main loss metric for training |
| **MAE** | Intuitive error in years |
| **R² Score** | Variance explained by the model |
| **KS Test** | Train-test distribution similarity check |
| **Visualization** | Actual vs. predicted age plots for error analysis |

---

## 📌 Model Highlights

### ✅ Modeling Workflow Enhancements

- 🧠 **Advanced ML Libraries Configured**:  
  ✅ XGBoost (GPU)  
  ✅ CatBoost (GPU)  
  ⚠️ LightGBM (CPU fallback on Colab)  

- 🛠️ **Preprocessing Optimization**:
  - Tested multiple scaling techniques (StandardScaler, RobustScaler, raw, minimal)
  - Final choice: **RobustScaler**, yielding lowest MSE (15.77) during early experimentation

---

### 🚀 Optimized Training Results

| Model                | Test MSE  | Test MAE | R² Score | Notes |
|---------------------|-----------|----------|----------|-------|
| **Enhanced RidgeCV** | **15.20** | **2.90** | 0.9277   | 🏆 Best individual model |
| Enhanced ElasticNet | 15.75     | 2.94     | 0.9251   | Expanded parameter tuning (375 combinations) |
| Ensemble (Ridge + Elastic + SVR) | 16.37     | 2.97     | 0.9221   | Final weighted ensemble |
| Quick SVR (GridSearchCV) | 37.72     | 4.39     | 0.8206   | Weakest individual performance |

🏅 **Final Model Selected**: `Enhanced RidgeCV`  
📈 **Performance**: MAE = **2.90 years**, MSE = **15.20**

---

### ⚙️ Minimal vs Enhanced Comparison

| Model              | Minimal MSE | Minimal MAE | Enhanced MSE | Enhanced MAE |
|-------------------|-------------|-------------|---------------|---------------|
| ElasticNet        | 19.95       | 3.31        | 15.75         | 2.94          |
| RidgeCV           | 22.10       | 3.55        | 15.20         | 2.90          |
| SVR (GridSearch)  | 30.75       | 3.77        | 37.72         | 4.39          |
| Simple Ensemble   | 19.41       | 3.16        | 16.37         | 2.97          |

---

### 🧠 Grid Search Optimization

- ✅ SVR GridSearchCV completed in **6.95 seconds** (48 combinations)  
- ✅ Enhanced ElasticNetCV: 375 combinations, runtime **~6 minutes**  
- ✅ Enhanced RidgeCV: 50 alpha values, runtime **~2 minutes**

---

### 📚 Benchmark Comparison: Methylation Clocks

| Model              | Reported MAE |
|--------------------|--------------|
| Horvath (2013)     | ~3.6 years   |
| Hannum (2013)      | ~4.9 years   |
| PhenoAge (2018)    | ~2.9 years   |
| **Our Model (Train)**  | **2.90 years** |
| **Our Model (Eval)**   | **3.90 years** |
| GrimAge (2019)     | ~2.3 years   |
| DunedinPACE (2022) | ~3.1 years   |

📊 The final model achieved a **Mean Absolute Error (MAE) of 2.90** on the training set and **3.90** on the evaluation/test set — comparable to established biological clocks like Horvath and PhenoAge, despite being trained and deployed with significantly **faster runtime and simpler architecture** using only blood-based CpG features.

---

### 📦 Output Summary

- ✅ Total Training Time: **24.5 seconds** (ensemble phase)  
- ✅ Final Test MAE: **3.90 years**  
- ✅ Final Test MSE: **15.20**  
- ✅ Model saved at: `/content/final_optimized_results.joblib`  
- ✅ Ensemble predictions and SHAP diagnostics exported  
- ✅ Memory optimized with `float32`, reducing footprint by **50%**

---

###🧪 Previously Used and Discarded Models Due to Errors in Real-World Usage

Several earlier models were explored in initial stages but ultimately discarded due to underperformance, instability, or poor generalization to biological methylation data. While models like **RidgeCV (old)** and **ElasticNet (old)** showed promise, they lacked consistent accuracy when evaluated under real-world preprocessing and cross-dataset conditions. Tree-based models such as **Random Forest**, **XGBoost**, and **Gradient Boosting Regressor** struggled with the high-dimensional CpG feature space, often overfitting or failing to capture subtle methylation-age relationships. Likewise, ensemble models and SVR (old) variants underperformed on actual test data despite strong internal scores. These models were phased out in favor of more interpretable and biologically robust linear models with enhanced preprocessing and tuning.

---

## 🧩 Applications & Impact

- 🧬 Research: Validate biological clocks and lifestyle effects  
- 🧠 Precision Health: Screen for accelerated aging or early-onset disease  
- 🧾 Forensics: Predict unknown individual’s age from blood sample  
- 🧮 Actuarial Science: Add biological age to health insurance models  

---

## 🔮 Future Enhancements

- 🔁 Ensemble modeling with Ridge + GB + CatBoost  
- 📚 SHAP/LIME explainability of CpG features  
- ⏳ Longitudinal analysis of aging trajectories  
- 🧫 Generalization across tissue types or populations  

---

> 📂 _Clone the repo, explore the notebook, and run the pipeline to dive into molecular aging prediction with machine learning._

---
