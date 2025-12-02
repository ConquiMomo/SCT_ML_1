\# SCT\_ML\_1



\## Internship task 1 — Machine learning (linear regression)



\### Project: Improved house price prediction model



This repository contains my solution for Task 1 of the Skillcraft Technology Machine Learning Internship. The goal is to build a regression model that predicts house prices using meaningful real‑estate features and a clean, reproducible workflow.



---



\## Repository structure



---



\## Steps performed

1\. Data loading — read train.csv and test.csv from ./data.

2\. Feature selection — OverallQual, GrLivArea, GarageCars, GarageArea, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd.

3\. Data cleaning — median imputation for missing numeric values.

4\. Feature scaling — StandardScaler on all selected features.

5\. Model training — scikit‑learn LinearRegression.

6\. Evaluation — RMSE, R², and error as % of median price.

7\. Submission — improved\_submission.csv with Id and SalePrice predictions.

8\. Analysis — plots: actual vs predicted, residuals, feature importance, price distribution.



---



\## Results

\- Training RMSE: (fill your value here)

\- Training R²: (fill your value here)

\- Median house price: (fill your value here)

\- Average error (% of median): (fill your value here)



Outputs:

\- improved\_submission.csv — test predictions

\- model\_analysis.png — model diagnostics



---



\## How to run

1\. Ensure data files exist:

&nbsp;  - Place `train.csv` and `test.csv` in `./data/`.

2\. Install dependencies:

&nbsp;  ```bash

&nbsp;  pip install pandas numpy matplotlib scikit-learn

