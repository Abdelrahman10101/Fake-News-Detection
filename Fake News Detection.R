#----------------------
#1-Libraries
#----------------------

library(tm)
#install.packages("textstem")
library(textstem)
#install.packages("SparseM")
library(SparseM)
library(text2vec)
library(Matrix)
library(caret)
#install.packages("yardstick")
library(yardstick)
library(nnet)
library(rpart)
library(e1071)
library(class)
#install.packages("randomForest")
library(randomForest)
#install.packages("glmnet")
library(glmnet)
library(ggplot2)
#----------------------
#2-preprocessing
#----------------------

# Load the data from CSV
news <- read.csv("news.csv")

# Check the dimensions of the dataset
dim(news)

# Check for missing values in each column
colSums(is.na(news))

# Dropping null values from the data frame
news_clean <- na.omit(news)

# Display the structure of the cleaned data
str(news_clean)

# Display the structure of the 'label' column before conversion
str(news_clean$label)

# Convert 'label' column to factor type
news_clean$label <- factor(news_clean$label)

# Display the structure of the 'label' column after conversion
str(news_clean$label)

# Get unique levels in the 'label' column
unique_levels <- levels(news_clean$label)
print(unique_levels)

# Copy the cleaned data to a new variable for encoding
news_clean_encoding <- news_clean

# Encoding 'label' column: F/FAKE to 0, R/REAL to 1
news_clean_encoding$label <- ifelse(news_clean$label %in% c("F", "FAKE"), 0, 1)

# Get unique levels in the encoded 'label' column
print(factor(news_clean_encoding$label))
str(news_clean_encoding$label)

stopwords("en")
# Custom function to remove URLs
removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)

# Custom function for lemmatization
lemmatizeText <- function(text) {
  lemmatized_text <- lemmatize_words(text)
  return(lemmatized_text)
}

# Create a corpus
corpus <- Corpus(VectorSource(news_clean_encoding$text))

# Preprocessing steps
corpus <- tm_map(corpus, content_transformer(tolower)) # Lowercasing
corpus <- tm_map(corpus, content_transformer(function(x) gsub("[^a-zA-Z]", " ", x)))#Replace non-alphabetic characters
corpus <- tm_map(corpus, content_transformer(removeURL)) # URL Removal
#corpus <- tm_map(corpus, content_transformer(splitContractions)) # Split Contractions
corpus <- tm_map(corpus, content_transformer(lemmatizeText)) # Lemmatization using textstem
corpus <- tm_map(corpus, removeWords, stopwords("en")) # Stop Word Removal


# Create Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
dim(dtm)  # Check dimensions of the DTM


# Convert DTM to a matrix
dtm_matrix <- as.matrix(dtm)
dim(dtm_matrix)

#----------------------
#3-Feature Selection
#----------------------

# Assuming dtm_matrix contains Document-Term Matrix and labels contain target labels
selected_features <- caret::nearZeroVar(dtm_matrix)
dtm_selected <- dtm_matrix[, selected_features]
dim(dtm_selected)

# Apply Chi-Square test
chi_squared <- apply(dtm_selected, 2, function(x) chisq.test(x, news_clean_encoding$label)$statistic)

# Get top K features
top_k <- 495  # Choose desired number of features
top_features <- names(sort(chi_squared, decreasing = TRUE))[1:top_k]
selected_dtm <- dtm_selected[, top_features]

# Data with target
final_dtm <- cbind(selected_dtm, label = news_clean_encoding$label)

# Convert final_dtm to Data Frame
final_dtm_df <- as.data.frame(final_dtm)

#----------------------
#4-Modeling
#----------------------

set.seed(123)  # Setting seed for reproducibility
train_indices <- sample(nrow(final_dtm_df), 0.7 * nrow(final_dtm_df))  # 70% train, 30% test
train_data <- final_dtm_df[train_indices, ]
test_data <- final_dtm_df[-train_indices, ]

#----------------------
#5-Logistic Regression
#----------------------

# Train the logistic regression model
model <- glm(label ~ ., data = train_data, family = binomial)

# Predictions on the test and train set
predictions_test <- predict(model, newdata = test_data, type = "response")
predictions_train <- predict(model, newdata = train_data, type = "response")

# Convert predicted probabilities to class labels (0 or 1)
predicted_classes_test <- ifelse(predictions_test > 0.5, 1, 0)
predicted_classes_train <- ifelse(predictions_train > 0.5, 1, 0)

# Calculate accuracy
accuracy_test <- mean(predicted_classes_test == test_data$label)
print(paste("Accuracy:", accuracy_test))

accuracy_train <- mean(predicted_classes_train == train_data$label)
print(paste("Accuracy:", accuracy_train))

#----------------------
#6-Decision Tree
#----------------------

# Train the decision tree model
model_tree <- rpart(label ~ ., data = train_data, method = "class")

# Predictions on the test  and train set
predictions_tree_test <- predict(model_tree, newdata = test_data, type = "class")
predictions_tree_train <- predict(model_tree, newdata = train_data, type = "class")

# Calculate accuracy for decision tree
accuracy_tree_test <- mean(predictions_tree_test == test_data$label)
print(paste("Decision Tree Accuracy:", accuracy_tree_test))

accuracy_tree_train <- mean(predictions_tree_train == train_data$label)
print(paste("Decision Tree Accuracy:", accuracy_tree_train))

#----------------------
#6-Naive Bayes
#----------------------

# Train the Naive Bayes model
nb_model <- naiveBayes(label ~ ., data = train_data)

# Predictions on the test and train set
nb_predictions_test <- predict(nb_model, newdata = test_data)
nb_predictions_train <- predict(nb_model, newdata = train_data)

# Calculate accuracy
nb_accuracy_test <- mean(nb_predictions_test == test_data$label)
print(paste("Accuracy:", nb_accuracy_test))

nb_accuracy_train <- mean(nb_predictions_train == train_data$label)
print(paste("Accuracy:", nb_accuracy_train))

#----------------------
#7-Confusion Matrix
#----------------------

# Create a confusion matrix for logistic regression
conf_matrix <- table(Actual = test_data$label, Predicted = predicted_classes_test)
conf_matrix

# Calculate precision, recall, and F1-score
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1-Score:", f1_score))

#----------------------
#8-LASSO regularization
#----------------------

# Prepare the data
train_matrix <- as.matrix(train_data[, -ncol(train_data)])  # Exclude the label column
test_matrix <- as.matrix(test_data[, -ncol(test_data)])    # Exclude the label column

train_labels <- as.numeric(train_data$label)
test_labels <- as.numeric(test_data$label)

# Fit logistic regression with Lasso regularization
lasso_model <- glmnet(train_matrix, train_labels, family = "binomial", alpha = 1)  # alpha = 1 for Lasso

# Make predictions
predictions_lasso_test <- predict(lasso_model, newx = test_matrix, type = "response")
predictions_lasso_train <- predict(lasso_model, newx = train_matrix, type = "response")

# Convert predicted probabilities to class labels (0 or 1)
predicted_classes_lasso_test <- ifelse(predictions_lasso_test > 0.5, 1, 0)
predicted_classes_lasso_train <- ifelse(predictions_lasso_train > 0.5, 1, 0)

# Calculate accuracy
accuracy_lasso_test <- mean(predicted_classes_lasso_test == test_labels)
print(paste("Accuracy:", accuracy_lasso_test))

accuracy_lasso_train <- mean(predicted_classes_lasso_train == train_labels)
print(paste("Accuracy:", accuracy_lasso_train))

#----------------------
#9-Visualization 
#----------------------

# Load the data from CSV
LR_extra <- read.csv("LR_extra.csv")
LR <- read.csv("LR.csv")
DT <- read.csv("DT.csv")
NB <- read.csv("NB.csv")

# Plotting bar graph for LR data using ggplot2
ggplot(LR_extra, aes(x = as.factor(rounded.accuracy), y = number.of.features )) +
  geom_bar(stat = "identity",fill = "red") +
  labs(x = "Accuracy", y = "Number of Features") +
  scale_y_continuous(breaks = seq(0, 3000, by = 100)) 
theme_minimal()

# Plotting the graph
plot(LR$number.of.features, LR$accuracy, type = "o", col = "red", ylim = range(LR$accuracy, DT$accuracy, NB$accuracy), 
     xlab = "Number of Features", ylab = "Accuracy")
lines(DT$number.of.features, DT$accuracy, type = "o", col = "blue")
lines(NB$number.of.features, NB$accuracy, type = "o", col = "green")
legend("bottomright", legend = c("LR", "DT", "NB"), col = c("red", "blue", "green"), lty = 1)


# Adding a column to identify the model
LR$model <- "LR"
DT$model <- "DT"
NB$model <- "NB"

# Combining data frames
combined_df <- rbind(LR, DT, NB)

# Plotting bar graph using ggplot2
ggplot(combined_df, aes(x = as.factor(number.of.features), y = accuracy, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Number of Features", y = "Accuracy") +
  scale_fill_manual(values = c("blue","red", "green")) +
  theme_minimal()
# bar graph for logistic regression
ggplot(LR, aes(x = as.factor(number.of.features), y = accuracy, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Number of Features", y = "Accuracy") +
  theme_minimal()
# bar graph for decision tree
ggplot(DT, aes(x = as.factor(number.of.features), y = accuracy, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Number of Features", y = "Accuracy") +
  theme_minimal()
# bar graph for naive bayes
ggplot(NB, aes(x = as.factor(number.of.features), y = accuracy, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Number of Features", y = "Accuracy") +
  theme_minimal()

# decision tree 
library(rpart)
#install.packages(rpart.plot)
library(rpart.plot)
rpart.plot(model_tree)
library(rattle)
fancyRpartPlot(model_tree)

#plotting histograms 
hist(LR$number.of.features, 
     main = "Histogram LOGISTIC REGRESSION", 
     xlab = "NUMBER OF FEATURES", 
     ylab = "", 
     breaks = "Sturges")

hist(LR$accuracy, 
     main = "Histogram LOGISTIC REGRESSION", 
     xlab = "ACCURACY", 
     ylab = "", 
     breaks = "Sturges")

hist(NB$number.of.features, 
     main = "Histogram NAIVE BAYES", 
     xlab = "NUMBER OF FEATURES", 
     ylab = "", 
     breaks = "Sturges")

hist(NB$accuracy, 
     main = "Histogram NAIVE BAYES", 
     xlab = "ACCURACY ", 
     ylab = "", 
     breaks = "Sturges")

hist(DT$number.of.features, 
     main = "Histogram DECISION TREE", 
     xlab = "NUMBER OF FEATURES", 
     ylab = "", 
     breaks = "Sturges")

hist(DT$accuracy, 
     main = "Histogram DECISION TREE", 
     xlab = "ACCURACY", 
     ylab = "", 
     breaks = "Sturges")


#Get top K features
top_k_visualization <- 10  # Choose desired number of features
top_features_visualization <- names(sort(chi_squared, decreasing = TRUE))[1:top_k_visualization]
selected_dtm_visualization <- dtm_selected[, top_features_visualization]

# Convert selected_dtm to Data Frame
final_dtm_df_visulization <- as.data.frame(selected_dtm_visualization)
view(final_dtm_df_visulization)
# Sum of each column
sum_per_column <- colSums(final_dtm_df_visulization, na.rm = TRUE)
view(sum_per_column)

pie(sum_per_column, 
    labels = names(sum_per_column), 
    col = rainbow(length(sum_per_column)),  # Optional color palette
    main = "PIE CHART MOST REPEATED WORDS")



