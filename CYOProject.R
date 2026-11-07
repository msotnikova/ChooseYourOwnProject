# Install and load the necessary packages
if (!require(tidyverse)) {
  install.packages("tidyverse")
  library(tidyverse)
}

if (!require(caret)) {
  install.packages("caret")
  library(caret)
}

if (!require(RKaggle)) {
  install.packages("RKaggle")
  library(RKaggle)
}

if (!require(randomForest)) {
  install.packages("randomForest")
  library(randomForest)
}

# Import directly into a tibble
census_data <- RKaggle::get_dataset("uciml/adult-census-income")

# Examine the data
print(head(census_data))

# Remove any row that contains a "?", convert categorical variables to factors, and save to a new tibble
census_data_clean <- census_data %>%
  mutate(across(where(is.character), str_trim)) %>%
  mutate(across(where(is.character), ~ na_if(.x, "?"))) %>%
  drop_na() %>%
  mutate(across(where(is.character), as.factor))

# Visualize the data
income_plot <- ggplot(census_data_clean, aes(x = income)) +
  geom_bar() +
  labs(
    title = "Distribution of Annual Income",
    x = "Income Category",
    y = "Number of Observations"
  )

print(income_plot)

# Create training set and testing set
set.seed(170)
train_index <- createDataPartition(y = census_data_clean$income, times = 1, p = 0.8, list = FALSE)
train_set <- census_data_clean[train_index, ]
test_set <- census_data_clean[-train_index, ]

# Separate predictors from the outcome
train_x <- train_set %>% select(-income)
test_x <- test_set %>% select(-income)

# Create dummy-variable transformation using training predictors only
dummy_model <- dummyVars(~ ., data = train_x, fullRank = FALSE)

# Apply transformation to training and testing sets
train_predictors <- predict(dummy_model, newdata = train_x)
test_predictors <- predict(dummy_model, newdata = test_x)

# Standardize training set
preprocess_model <- preProcess(train_predictors, method = c("center", "scale"))
train_predictors_scaled <- predict(preprocess_model, train_predictors)

# Standardize testing set using the same parameters
test_predictors_scaled <- predict(preprocess_model, test_predictors)

# Store training set and testing set outcomes
train_income <- as.factor(train_set$income)
test_income <- factor(test_set$income, levels = levels(train_income))

# Define cross-validation
train_control <- trainControl(method = "cv", number = 3)

# Train KNN model
set.seed(170)
knn_model <- train(
  x = train_predictors_scaled,
  y = train_income,
  method = "knn",
  trControl = train_control,
  tuneGrid = data.frame(k = c( 5, 9, 13))
)

# Display results
print(knn_model)

# Display optimal k
print(knn_model$bestTune)

# Predict income using the testing dataset
y_hat_knn <- predict(knn_model, newdata = test_predictors_scaled)

# Generate confusion matrix
cm <- confusionMatrix(data = y_hat_knn, reference = test_income)

# Display confusion matrix
print(cm)

# Display classification accuracy
print(cm$overall["Accuracy"])

# Define parameters for Random Forest
set.seed(170)
control <- trainControl(method="cv", number = 3)
grid <- data.frame(mtry = c(5, 10, 20))

train_rf <-  train(train_predictors_scaled, train_income, 
                   method = "rf", 
                   ntree = 100,
                   trControl = control,
                   tuneGrid = grid)

# Display Random Forest cross-validation results
print(train_rf)

# Display optimal mtry
print(train_rf$bestTune)

# Display detailed tuning results
print(train_rf$results)

# Predict income using the testing dataset
y_hat_rf <- predict(train_rf, test_predictors_scaled)

# Generate confusion matrix
cm <- confusionMatrix(y_hat_rf, test_income)

# Display confusion matrix
print(cm)

# Display classification accuracy
print(cm$overall["Accuracy"])

# Store results
cm_knn <- confusionMatrix(y_hat_knn, test_income)
cm_rf <- confusionMatrix(y_hat_rf, test_income)

model_results <- tibble(
  Model = c("KNN", "Random Forest"),
  Accuracy = c(
    unname(cm_knn$overall["Accuracy"]),
    unname(cm_rf$overall["Accuracy"])
  )
)

# Display results
print(model_results)