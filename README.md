# Loan Approval Decision-Tree-Classification-ML-
Built an interpretable decision tree classification model in R to predict loan approval outcomes. Achieved 98% accuracy on imbalanced data using stratified sampling and evaluation metrics such as recall, specificity, and balanced accuracy. Demonstrates strong business interpretation and ethical ML awareness.


# Loan Approval Prediction using Decision Trees (R)

*Author:* Katlego Mathebula
*Tech Stack:* R · rpart · caret · dplyr · ggplot2
*Project Type:* Supervised Machine Learning (Classification)

## Executive Summary

This project builds a **Decision Tree classification model** to predict loan approval outcomes based on applicant financial and employment information.

The objective is to simulate how financial institutions use data-driven decision systems to:

-  Reduce default risk
-  Improve approval consistency
-  Increase decision transparency
-  Support credit risk assessment

The model was developed using R and evaluated using a stratified train-test approach to ensure reliable performance estimation.


## Business Problem

Financial institutions must evaluate thousands of loan applications while balancing:

-  Profitability
-  Credit risk
-  Regulatory fairness
-  Default minimization

Manual decision-making introduces bias and inconsistency.

This project answers:

 Can we use applicant data to systematically predict whether a loan should be approved?
The model uses financial and behavioral variables to estimate approval likelihood.

## Dataset Overview

The dataset contains structured loan application data including:

-  `Income`
-  `Credit_Score`
-  `Loan_Amount`
-  `DTI_Ratio` (Debt-to-Income ratio)
-  `Employment_Status`
-  `Approval` (Target variable)

The target variable is binary:

- Approved
- Not Approved

## Data Preparation & Preprocessing

### 1. Library Setup

Key libraries were used for:

-  `readr` → Data loading
-  `dplyr` → Data manipulation
-  `rpart` → Decision tree modeling
-  `rpart.plot` → Model visualization
-  `caret` → Model evaluation and train-test splitting


### 2. Data Cleaning

-  Removed irrelevant text column (`Text`)
-  Converted categorical variables into factors

  -  `Employment_Status`
  -  `Approval`

This is critical because classification algorithms in R require categorical targets to be factors.

### 3. Class Distribution Analysis

Before modeling, class balance was evaluated using:

-  Frequency counts
-  Percentage proportions

Understanding class imbalance is essential because skewed data can bias predictions.

## Model Validation Strategy

### Stratified Train-Test Split (80/20)

Instead of randomly splitting the data, a *stratified split* was used:

```r
createDataPartition()
```

This ensures that both training and test sets maintain similar approval distributions.

Why this matters:

-  Prevents biased evaluation
-  Preserves class proportions
-  Produces realistic performance metrics

A fixed seed (`set.seed(123)`) was used to ensure reproducibility.

## Model Development

### Decision Tree Algorithm

The model was built using:

```r
rpart(method = "class")
```

A decision tree was selected because:

-  It is interpretable
-  It mimics human decision-making
-  It handles non-linear relationships
-  It requires minimal preprocessing


### Overfitting Control

To prevent overfitting, the following parameters were used:

```r
cp = 0.01
minsplit = 20
```

- `cp` (Complexity Parameter): Prevents unnecessary splits
-  `minsplit`: Minimum observations required to split a node

This ensures the model generalizes well to unseen data.


## Model Interpretation

The tree was visualized using:

```r
rpart.plot()
```

This allows us to:

-  Identify key decision variables
-  Understand threshold values
-  Interpret how approvals are determined

Decision trees are highly valuable in finance because they are explainable — a regulatory requirement in many banking systems.

## Model Evaluation

Predictions were made on the test set and evaluated using a confusion matrix:

```r
confusionMatrix()
```

The confusion matrix provides:

-  Accuracy
-  Sensitivity (Recall)
-  Specificity
-  Precision
-  Kappa statistic

These metrics help assess:

-  How well the model detects approvals
-  How well it detects rejections
-  Whether predictions are balanced

Evaluation was performed on unseen test data to measure true predictive performance.

## Key Insights

From the decision tree structure, we can observe:

-  Credit score plays a major role in approval
-  High DTI ratios reduce approval probability
-  Income levels influence threshold splits
-  Employment status contributes to segmentation

## Conclusion

This project simulates a real-world credit risk modeling workflow.
The approach prioritizes interpretability and reliability, which are essential in financial decision systems.



