# LOAN-APPROVAL-ML-DECISION-TREE-CLASSIFICATION
This project demonstrates how to build a Decision Tree model in R to predict loan approvals based on applicant information. It includes data preprocessing, model training, visualization, and evaluation.
## 1. Project Overview

The goal of this project is to predict whether a loan will be approved (Approval) using a dataset containing financial and employment details of applicants. A Decision Tree classifier is used due to its interpretability and ease of visualization.

## 2. Libraries Required

The following R packages are used in this project:

library(readr)       # Reading CSV files
library(rpart)       # Decision Tree modeling
library(rpart.plot)  # Tree visualization
library(caret)       # Data splitting, evaluation metrics
library(dplyr)       # Data manipulation
library(ggplot2)     # Data visualization

##  3. Dataset

The dataset loan_data.csv includes the following key variables:

Income – Applicant's monthly income
Credit_Score – Applicant's credit score
Loan_Amount – Requested loan amount
DTI_Ratio – Debt-to-Income ratio
Employment_Status – Employment status of the applicant

Approval – Target variable: Yes or No

The Text column is removed during preprocessing as it is not relevant for modeling.

## 4.  Data Preprocessing

Remove irrelevant columns (Text).

Convert categorical variables (Employment_Status and Approval) to factors.

Check class balance for the target variable:

table(loan_data$Approval)
prop.table(table(loan_data$Approval)) * 100


Stratified train-test split (80% train, 20% test) to maintain class balance.

## 5. Model Training

A Decision Tree classifier is trained on the following features:

* Income

* Credit_Score

* Loan_Amount

* DTI_Ratio

* Employment_Status

Model settings:

cp = 0.01 → Complexity parameter to avoid overfitting

minsplit = 20 → Minimum observations required to split a node

dt_model <- rpart(
  Approval ~ Income + Credit_Score + Loan_Amount + DTI_Ratio + Employment_Status,
  data = train_data,
  method = "class",
  control = rpart.control(cp = 0.01, minsplit = 20)
)

 ## 6. Decision Tree Visualization

The trained tree can be visualized for better interpretability:

rpart.plot(
  dt_model,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE
)


The plot shows decision rules at each node and the proportion of approvals/rejections.

##  7. Model Evaluation

Predictions are made on the test set, and a confusion matrix is used to evaluate model performance:

test_pred <- predict(dt_model, newdata = test_data, type = "class")

confusionMatrix(test_pred, test_data$Approval)


The confusion matrix provides accuracy, sensitivity, specificity, and other key metrics.

## 8.  Summary

This project demonstrates a complete workflow for predicting loan approvals using a Decision Tree in R:
Data loading and preprocessing
Train-test split
Decision Tree model training
Visualization of the tree
Model evaluation using a confusion matrix
This framework can be extended to include other machine learning algorithms or feature engineering for better predictive performance.

## 9. Files

loan_data.csv – Input dataset

Author Katlego Mathebula
loan_approval_decision_tree.R – Full R script for model training and evaluation
