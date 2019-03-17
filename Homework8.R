########################################################################
## Title: Homework 8
## Author: Jonah Witt
## Description: This file analyzes customer reviews and uses supervised
## learning to classify lies. 
########################################################################

# Load Libraries
library(tidytext)
library(reshape2)
library(naivebayes)
library(tm)
library(e1071)
library(randomForest)
library(caret)
library(rpart)
library(rattle)
library(rpart.plot)
library(ggplot2)
library(arules)

# Set working directory
setwd("~/Desktop/IST 707")

# Load in the data
dat <- readLines("deception_data_converted_final.csv")

########################################################################
## Data Cleaning
########################################################################

#remove header row
dat <- dat[c(2:93)]
#remove empty reviews
dat <- dat[-c(83:84)]


# Initialize Empty Vectors to Store lie, sentiment, and review
lie <- vector(mode = "character", length = length(dat))
sentiment <- vector(mode = "character", length = length(dat))
review <- vector(mode = "character", length = length(dat))

# initialize i to iterate over lines
i = 1;

# iterate over line and extract the lie, sentiment, and review values
for(line in dat){
  lie[i] <- substr(line[1], 1, 1)
  sentiment[i] <- substr(line[1], 3, 3)
  review[i] <- substr(line[1], 5, nchar(line))
  i = i + 1
}

# combine all three vectors into a df
df <- data.frame(lie, sentiment, review)

# get corpus
corpus <- VCorpus(VectorSource(df$review))

#remove punctuation
cleanCorpus <- tm_map(corpus, removePunctuation)

#Remove stop words
cleanCorpus <- tm_map(cleanCorpus, removeWords, stopwords("english"))

#Remove numbers
cleanCorpus <- tm_map(cleanCorpus, removeNumbers)

#Remove whitespace
cleanCorpus <- tm_map(cleanCorpus, stripWhitespace)

#View corpus as term document matrix
DTM <- DocumentTermMatrix(cleanCorpus)

# Normalize DTM
normalizedDTM <- DocumentTermMatrix(cleanCorpus, control = list(weighting = weightTfIdf, stopwords = TRUE))

#Get matrix with counts of key words in each document
wordCounts <- tidy(normalizedDTM)
wordCounts <- dcast(data = wordCounts,formula = document~term,fun.aggregate = sum)
rownames(wordCounts) <- wordCounts$document
wordCounts <- wordCounts[-1]
wordCounts[1:10]

# Scale 
scaledWordCounts <- data.frame(scale(wordCounts))

# Sort df on row names
scaledWordCounts <- scaledWordCounts[ order(as.numeric(row.names(scaledWordCounts))), ]

# Add lie Detection
lieDetection <- scaledWordCounts
lieDetection$lie <- as.factor(lie)

# Add sentiment
sentimentAnalysis <- scaledWordCounts
sentimentAnalysis$sentiment <- as.factor(sentiment)

########################################################################
## Lie Detection
########################################################################

# Split the data
set.seed(474)
#Split data set
trainRows <- sample(1:nrow(lieDetection),0.80*nrow(lieDetection))
train <- lieDetection[trainRows, ]
test <- lieDetection[-trainRows, ]

########################################################################
## Random Forest
########################################################################

# Set up RF
rf <- randomForest(lie~. , data = train)
print(rf)
# Make predictions
predictions <- predict(rf, test) 
cv = (table(predictions, test$lie))
print(cv)
accuracy <- ((cv[1,1] + cv[2,2])/length(test$lie))
print(accuracy)

########################################################################
## Support Vector Machine
########################################################################

# Set up SVM
clf <- svm(lie~., data=train, kernel="linear", scale=FALSE)
print(clf)

# Get Predictions
(predictions <- predict(clf, test, type="class"))

# Get confusion matrix
(cv <- table(predictions, test$lie))
print(cv)
accuracy <- ((cv[1,1] + cv[2,2])/length(test$lie))
print(accuracy)

########################################################################
## Decision Tree
########################################################################

#create a function to display results of decision tree
get_results <- function(clf){
  #get predictions from the tree
  predictions <- predict(clf, test, type = "class")
  #visualize the tree
  fancyRpartPlot(clf)
  #Get cross validation table
  cv <- table(predictions, test$lie)
  print(cv)
  #Get accuracy of model
  accuracy <- ((cv[1,1] + cv[2,2])/length(predictions))
  print(accuracy)
}

#Create a function to prune tree
prune_tree <- function(clf){
  ptree<- prune(clf, cp= clf$cptable[which.min(clf$cptable[,"xerror"]),"CP"])
  get_results(ptree)
}

formula <- formula(lie~., data = train)
minSplit <- 1
myCp = -1

#Create a function to easily test different tunings on the tree
get_tree <- function(formula, minSplit, myCp){
  myClf <- rpart(formula, data = train, method = "class", control = c(minsplit = minSplit, cp = myCp))
  print("Full Tree Results")
  get_results(clf = myClf)
  print("Pruned Tree Results")
  prune_tree(clf = myClf)
}

#fit the decision tree and view results
get_tree(formula, minSplit, myCp)

########################################################################
## Naive Bayes
########################################################################

NB_object<- naive_bayes(lie~., data=train)
NB_prediction<-predict(NB_object, test[,-1335], type = c("class"))
head(predict(NB_object, test[,-1335], type = "class"))
table(NB_prediction,test$lie)

########################################################################
## k-Nearest Neighbor
########################################################################

# Set k
train$lie <- as.numeric(train$lie)
test$lie <- as.numeric(test$lie)
k <- 4
# Fit the model
kNN_fit <- class::knn(train=train, test=test, cl=train$lie, k = k, prob=TRUE)
print(kNN_fit)

## Check the classification accuracy
cv = (table(kNN_fit, test$lie))
print(cv)
accuracy <- ((cv[1,1] + cv[2,2])/length(test$lie))
print(accuracy)

########################################################################
## Sentiment Analysis
########################################################################

# Split the data
set.seed(474)
#Split data set
trainRows <- sample(1:nrow(sentimentAnalysis),0.80*nrow(sentimentAnalysis))
train <- sentimentAnalysis[trainRows, ]
test <- sentimentAnalysis[-trainRows, ]

########################################################################
## Random Forest
########################################################################

# Set up RF
rf <- randomForest(sentiment~. , data = train)
print(rf)
# Make predictions
predictions <- predict(rf, test) 
cv = (table(predictions, test$sentiment))
print(cv)
accuracy <- ((cv[1,1] + cv[2,2])/length(test$sentiment))
print(accuracy)

########################################################################
## Support Vector Machine
########################################################################

# Set up SVM
clf <- svm(sentiment~., data=train, kernel="linear", scale=FALSE)
print(clf)

# Get Predictions
(predictions <- predict(clf, test, type="class"))

# Get confusion matrix
(cv <- table(predictions, test$sentiment))
print(cv)
accuracy <- ((cv[1,1] + cv[2,2])/length(test$sentiment))
print(accuracy)

########################################################################
## Decision Tree
########################################################################

#create a function to display results of decision tree
get_results <- function(clf){
  #get predictions from the tree
  predictions <- predict(clf, test, type = "class")
  #visualize the tree
  fancyRpartPlot(clf)
  #Get cross validation table
  cv <- table(predictions, test$sentiment)
  print(cv)
  #Get accuracy of model
  accuracy <- ((cv[1,1] + cv[2,2])/length(predictions))
  print(accuracy)
}

#Create a function to prune tree
prune_tree <- function(clf){
  ptree<- prune(clf, cp= clf$cptable[which.min(clf$cptable[,"xerror"]),"CP"])
  get_results(ptree)
}

formula <- formula(sentiment~., data = train)
minSplit <- 1
myCp = -1

#Create a function to easily test different tunings on the tree
get_tree <- function(formula, minSplit, myCp){
  myClf <- rpart(formula, data = train, method = "class", control = c(minsplit = minSplit, cp = myCp))
  print("Full Tree Results")
  get_results(clf = myClf)
  print("Pruned Tree Results")
  prune_tree(clf = myClf)
}

#fit the decision tree and view results
get_tree(formula, minSplit, myCp)

########################################################################
## Naive Bayes
########################################################################

NB_object<- naive_bayes(sentiment~., data=train)
NB_prediction<-predict(NB_object, test[,-1335], type = c("class"))
head(predict(NB_object, test[,-1335], type = "class"))
table(NB_prediction,test$sentiment)


########################################################################
## k-Nearest Neighbor
########################################################################

# Set k
train$sentiment <- as.numeric(train$sentiment)
test$sentiment <- as.numeric(test$sentiment)
k <- 4
# Fit the model
kNN_fit <- class::knn(train=train, test=test, cl=train$sentiment, k = k, prob=TRUE)
print(kNN_fit)

## Check the classification accuracy
cv = (table(kNN_fit, test$sentiment))
print(cv)
accuracy <- ((cv[1,1] + cv[2,2])/length(test$sentiment))
print(accuracy)






