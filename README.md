# Aging Biomarkers-Methylation based Chronological Age Estimator
Understanding the biological basis of aging through epigenetic signatures using machine learning. This project leverages public DNA methylation datasets to develop predictive models for estimating chronological age from CpG site patterns.

---

## Data Sources:

-GSE40279 – Training dataset from NCBI GEO
-GSE157131 – Evaluation dataset from NCBI GEO
Please refer to the original GEO pages for authorship and licensing details.

---

## Why It Matters
DNA methylation-based biomarkers offer a powerful tool for understanding biological aging. Chronological age estimation from blood samples has critical applications in:
-**Forensic science**: Age prediction in unidentified samples.
-**Healthcare**: Early identification of accelerated aging or age-related diseases.
-**Research**: Understanding the epigenetic effects of environment and lifestyle.

## The Challenge
Current age estimation models vary in accuracy across datasets.
Accurate, robust models are needed to generalize well across populations.

## The Impact
Improved prediction models enable better diagnostics, scientific insights, and targeted interventions across clinical and population health domains.

---

## Key Techniques Used
-**CpG Site Variability Filtering**
Selected the top 10,000 most variable CpG sites based on variance analysis.

-**Bootstrap Feature Selection**
Applied repeated sampling and Ridge regression to identify stable age-associated CpG markers.

-**Lasso Regression for Feature Refinement**
Used LassoCV to further reduce CpG features by eliminating non-contributing coefficients.

-**Model Training & Evaluation**
Trained multiple regression models:
- ElasticNet
- Ridge Regression
- Gradient Boosting
- Support Vector Regression (SVR)
- Random Forest

---

## Performance Metrics
Evaluation based on:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Actual vs. Predicted Age Visualization

---

# Model Highlights 

- **Objective**: Predict chronological age using DNA methylation profiles (CpG site data).
- **Data Used**: GSE40279 (training/test), GSE157131 (external evaluation).
- **Feature Engineering**:
  - CpG variance filtering
  - Ridge-based bootstrap selection
  - LassoCV for final CpG feature refinement

---

## Model Selection Criteria

### **ElasticNet Regression**
Balances regularization and feature selection for high-dimensional CpG data.

### **Ridge Regression (Tuned)**
Robust to multicollinearity; cross-validated alpha enhances generalization.

### **Gradient Boosting (Tuned)**
Captures complex patterns; tuned for optimal depth and learning rate.

### **SVR**
Explores kernel-based modeling in high-dimensional spaces.

### **Stacking Regressor**
Leverages strengths of multiple models to improve accuracy.

---

## ElasticNet Regression
- Combines L1 & L2 regularization for sparse but stable modeling.
- Test MSE: 15.31
- Eval MAE: 9.37

## Ridge Regression (Tuned)
- Pure L2 regularization with alpha selection via cross-validation.
- Test MSE: 11.92 (best test score)
- Eval MAE: 12.60 (likely overfit)

## Gradient Boosting Regressor (Tuned)
- Ensemble of decision trees optimized for residuals.
- Test MSE: 36.60
- Eval MAE: 4.56 (best generalization)

## Support Vector Regressor (SVR)
- Kernel-based model; underperformed in this setting.
- Test MSE: 43.02

## Stacking Regressor 
- Meta-model combining ElasticNet, Ridge, and GB.
- Test MSE: 12.85 (good ensemble performance)

---

## Why Gradient Boosting (Tuned) Was Selected for Final Evaluation
- Achieved lowest MAE on the evaluation dataset.
- Handled high-dimensional feature space effectively.
- Benefited from hyperparameter tuning to reduce overfitting.
- Consistently aligned predicted ages with actual ages in visual diagnostics.
  
---

## Conclusion
This project developed a machine learning pipeline to estimate chronological age using DNA methylation data. After evaluating multiple regression models—including ElasticNet, SVR, and Gradient Boosting—the tuned Ridge Regression model demonstrated the best performance on test data (MSE: 11.91) and offered robust generalization with minimal overfitting. On the external evaluation set, Tuned Gradient Boosting achieved the best results (MAE: 4.56), indicating its strength in capturing complex, non-linear methylation-age relationships.  

---

## Future Work
- Ensemble Modeling: Combine top models like Ridge and GB for improved robustness.
- Biological Validation: Investigate biological relevance of top CpG markers.
- Longitudinal Data: Extend to predict biological aging over time or healthspan metrics.
- Cross-Tissue Generalization: Test model performance across tissues or ethnic cohorts.
- Explainability: Apply SHAP or LIME to interpret CpG contributions to age predictions.

---
