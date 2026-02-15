## 1. Adding all the required libraries 
library(readr)
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)
library(ggplot2)

## 2. Loading the data set
loan_data <- read_csv("loan_data.csv")

## 3. Viewing the data set
View(loan_data)

## 4. Removing the Text column 
loan_data<- loan_data%>%
  select(-Text)

colnames(load_data)


## 5. Convert Categorial variables into factors 
loan_data$Employment_Status<- as.factor(loan_data$Employment_Status)
loan_data$Approval<-as.factor(loan_data$Approval)

str(loan_data)
levels(loan_data$Employment_Status)

## 6. Checking Approval Balance 

table(loan_data$Approval) # Frequency counts
prop.table(table(loan_data$Approval))*100  #proportions (percentages)

## 7. Stratified Train-Test Split

set.seed(123)

# Creating a stratified split (Train 80% and test 20%)
train_index <-createDataPartition(
            loan_data$Approval,
            p=0.8,
            list=FALSE
)

train_data <- loan_data [train_index, ]
test_data <- loan_data[-train_index,   ]

# Checking balance in the train set
prop.table(table(train_data$Approval))*100

# Checking the balance in the test set
prop.table(table(test_data$Approval))*100


## 8. Train the Decision tree 

# Train decision tree model 
dt_model <- rpart(
  Approval ~ Income + Credit_Score + Loan_Amount + DTI_Ratio + Employment_Status,
  data=train_data,
  method="class",
  control = rpart.control(
    cp=0.01,        # Complexity parameter which prevents over fitting
    minsplit = 20  # this is the minimum obs to split
  )
)

## 9. Visualizing the Tree 

rpart.plot(
  dt_model,
  type =2,
  extra= 104,
  fallen.leaves = TRUE
)

## 10. Making predictions on test data

test_pred <- predict (
  dt_model,
  newdata = test_data,
  type = "class"
)

## 11. The confusion matrix, this are the core results

confusionMatrix(
  test_pred,
  test_data$Approval
)

##  12. END OF CODE !