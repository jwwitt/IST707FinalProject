########################################################################
## Title: Final Project
## Author: Jonah Witt, Taylor Moorman, Michael Morales
## Description: This file analyses economic data from the FIFA World Cup 
## from 1930 to 2018 
########################################################################

# Load Libraries
library(stringr)
library(e1071)
library(randomForest)
library(caret)
library(rpart)
library(rattle)
library(rpart.plot)
library(ggplot2)
library(arules)
library(kknn)
library(plyr)

# Set working directory
setwd("C:\\Users\\tmoorman\\Documents\\R\\IST707FinalProject-master")

# Load in the data
matchData <- read.csv("WorldCupMatches.csv")
WorldCupData <- read.csv("WorldCups.csv")

########################################################################
## Data Cleaning
########################################################################

# Remove ID variables from match data
matchData <- matchData[ , -which(names(matchData) %in% c("RoundID","MatchID"))]

# Remove Team Initials from match data
matchData <- matchData[ , -which(names(matchData) %in% c("Home.Team.Initials","Away.Team.Initials"))]

# Remove Date and only keep time in Datetime
matchData$Datetime <- str_extract(matchData$Datetime, '[0-9][0-9]:[0-9][0-9]')

# Change column names in match data
colnames(matchData) <- c("year", "time", "stage", "stadium", "city", "homeTeam", "hometeamGoals", "awayTeamGoals", 
                         "awayTeam", "winConditions", "attendance", "halfTimeHomeGoals", "halfTimeAwayGoals", 
                         "referee", "assistant1", "assistant2", "matchWin", "winPlace", "winWin")

# Check for NA in match data
summary(complete.cases(matchData))

# Remove Win Conditions Column from match data. Too many missing values.
matchData <- matchData[ , -which(names(matchData) %in% c("winConditions"))]

# Remove remaining records with missing values in match data
matchData <- matchData[complete.cases(matchData),]

# Get nationality of officials
i <- 1
for(row in matchData){
  matchData$refereeNationality[i] <- gsub("[\\(\\)]", "", regmatches(matchData$referee[i], 
                                                                  gregexpr("\\(.*?\\)", 
                                                                  matchData$referee[i]))[[1]])
  matchData$assistant1Nationality[i] <- gsub("[\\(\\)]", "", regmatches(matchData$assistant1[i], 
                                                                  gregexpr("\\(.*?\\)", 
                                                                  matchData$assistant1[i]))[[1]])
  matchData$assistant2Nationality[i] <- gsub("[\\(\\)]", "", regmatches(matchData$assistant2[i], 
                                                                  gregexpr("\\(.*?\\)", 
                                                                  matchData$assistant2[i]))[[1]])
  i <- i + 1
}

# convert officials to char type
matchData$referee <- as.character(matchData$referee)
matchData$assistant1 <- as.character(matchData$assistant1)
matchData$assistant2 <- as.character(matchData$assistant2)

# remove nationality from officials names
matchData$referee <- substr(matchData$referee,1,nchar(matchData$referee)-6)
matchData$assistant1 <- substr(matchData$assistant1,1,nchar(matchData$assistant1)-6)
matchData$assistant2 <-  substr(matchData$assistant2,1,nchar(matchData$assistant2)-6)

# Duplicate data frame to separate home and away teams
homeDF <- matchData
awayDF <- matchData

# Get team name for each game for home and away teams
homeDF$team <- homeDF$homeTeam
awayDF$team <- awayDF$awayTeam

# Get opponent name for each game for home and away teams
homeDF$opponent <- homeDF$awayTeam
awayDF$opponent <- awayDF$homeTeam

# Get home or away
homeDF$homeOrAway <- "home"
awayDF$homeOrAway <- "away"

# Remove home team and away team columns
homeDF <- homeDF[ , -which(names(homeDF) %in% c("homeTeam", "awayTeam"))]
awayDF <- awayDF[ , -which(names(awayDF) %in% c("homeTeam", "awayTeam"))]

# iterate over home df and get results
for(i in 1:length(homeDF$team)){
  if(homeDF$hometeamGoals[i] > homeDF$awayTeamGoals[i]){
    homeDF$result[i] <- "win"
  }else if(homeDF$hometeamGoals[i] < homeDF$awayTeamGoals[i]){
    homeDF$result[i] <- "loss"
  }else{
    homeDF$result[i] <- "draw"
  }
}

# iterate over away df and get results
for(i in 1:length(awayDF$team)){
  if(awayDF$awayTeamGoals[i] > awayDF$hometeamGoals[i]){
    awayDF$result[i] <- "win"
  }else if(awayDF$awayTeamGoals[i] < awayDF$hometeamGoals[i]){
    awayDF$result[i] <- "loss"
  }else{
    awayDF$result[i] <- "draw"
  }
}

# Rename goals columns
names(homeDF)[names(homeDF) == 'hometeamGoals'] <- 'goalsFor'
names(homeDF)[names(homeDF) == 'awayTeamGoals'] <- 'goalsAgainst'
names(homeDF)[names(homeDF) == 'halfTimeHomeGoals'] <- 'goalsForHalfTime'
names(homeDF)[names(homeDF) == 'halfTimeAwayGoals'] <- 'goalsAgainstHalfTime'

names(awayDF)[names(awayDF) == 'awayTeamGoals'] <- 'goalsFor'
names(awayDF)[names(awayDF) == 'hometeamGoals'] <- 'goalsAgainst'
names(awayDF)[names(awayDF) == 'halfTimeAwayGoals'] <- 'goalsForHalfTime'
names(awayDF)[names(awayDF) == 'halfTimeHomeGoals'] <- 'goalsAgainstHalfTime'

# combine homeDF and awayDF
df <- rbind(homeDF, awayDF)

# add empty host column to df
df$host <- ""

# add host nation to each world cup
for(i in 1:length(df$team)){
  if(df$year[i] == 1930){
    df$host[i] <- "Uruguay"
  }else if(df$year[i] == 1934){
    df$host[i] <- "Italy"
  }else if(df$year[i] == 1938){
    df$host[i] <- "France"
  }else if(df$year[i] == 1950){
    df$host[i] <- "Brazil"
  }else if(df$year[i] == 1954){
    df$host[i] <- "Switzerland"
  }else if(df$year[i] == 1958){
    df$host[i] <- "Sweden"
  }else if(df$year[i] == 1962){
    df$host[i] <- "Chile"
  }else if(df$year[i] == 1966){
    df$host[i] <- "England"
  }else if(df$year[i] == 1970){
    df$host[i] <- "Mexico"
  }else if(df$year[i] == 1974){
    df$host[i] <- "Germany"
  }else if(df$year[i] == 1978){
    df$host[i] <- "Argentina"
  }else if(df$year[i] == 1982){
    df$host[i] <- "Spain"
  }else if(df$year[i] == 1986){
    df$host[i] <- "Mexico"
  }else if(df$year[i] == 1990){
    df$host[i] <- "Italy"
  }else if(df$year[i] == 1994){
    df$host[i] <- "United States"
  }else if(df$year[i] == 1998){
    df$host[i] <- "France"
  }else if(df$year[i] == 2002){
    df$host[i] <- "Japan and South Korea"
  }else if(df$year[i] == 2006){
    df$host[i] <- "Germany"
  }else if(df$year[i] == 2010){
    df$host[i] <- "South Africa"
  }else if(df$year[i] == 2014){
    df$host[i] <- "Brazil"
  }else{
    df$host[i] <- "Russia"
  }
}

# add whether each country was host nation
for(i in 1:length(df$team)){
  if(df$host[i] == df$team[i]){
    df$isHostCountry[i] <- "yes"
  }else{
    df$isHostCountry[i] <- "no"
  }
}

# remove city and stadium columns from df (redundant with host attributes)
df <- df[ , -which(names(homeDF) %in% c("stadium", "city"))]

# get correct data types
df$year <- as.factor(df$year)
df$time <- as.factor(df$time)
df$referee <- as.factor(df$referee)
df$assistant1 <- as.factor(df$assistant1)
df$assistant2 <- as.factor(df$assistant2)
df$refereeNationality <- as.factor(df$refereeNationality)
df$assistant1Nationality <- as.factor(df$assistant1Nationality)
df$assistant2Nationality <- as.factor(df$assistant2Nationality)
df$homeOrAway <- as.factor(df$homeOrAway)
df$result <- as.factor(df$result)
df$winPlace <- as.factor(df$winPlace)
df$winWin <- as.factor(df$winWin)
df$host <- as.factor(df$host)
df$isHostCountry <- as.factor(df$isHostCountry)
df$result <- as.factor(df$result)

# remove unnecessary columns
df <- df[ , -which(names(df) %in% c("referee", "assistant1", "assistant2", "matchWin", "team", "opponent"))]

########################################################################
## Data Visualization
########################################################################

#Exploring trends in match results
plot(df$stage, df$attendance, col="red")
plot(df$isHostCountry, df$result) #much more likely to win if host country == yes
plot(df$isHostCountry, df$winPlace)
plot(df$isHostCountry, df$winWin)
plot(df$homeOrAway, df$result)
plot(df$result, df$goalsFor)

#Exploring Financial Results
plot(WorldCupData$year, WorldCupData$fifarev)
plot(WorldCupData$year, WorldCupData$hostcost)
plot(WorldCupData$year, WorldCupData$cost_inf)
plot(WorldCupData$year, WorldCupData$rev_inf)
plot(WorldCupData$year, WorldCupData$prizes)
WorldCupData$prizes_inf - WorldCupData$hostcost


########################################################################
## Association Rules
########################################################################

# get df for association rules
ruleDF <- df

# set all but attendance as factor
ruleDF$goalsAgainst <- as.factor(ruleDF$goalsAgainst)
ruleDF$goalsFor <- as.factor(ruleDF$goalsFor)
ruleDF$goalsAgainstHalfTime <- as.factor(ruleDF$goalsAgainstHalfTime)
ruleDF$goalsForHalfTime <- as.factor(ruleDF$goalsForHalfTime)

# discretize attendance variable
ruleDF$attendance <- as.factor(ifelse(ruleDF$attendance <= 30000,'low',ifelse(ruleDF$attendance <= 61381, 'average', 'high')))

#Get association rules
rules <- apriori(ruleDF, parameter = list(conf = 0.99, maxlen = 2), control = list(verbose=F))
arules::inspect(rules)

########################################################################
## k-Means Clustering
########################################################################
# copy df
dfOrig <- df

# get all numeric variables
df$year <- as.numeric(df$year)
df$time <- as.numeric(df$time)
df$stage <- as.numeric(df$stage)
df$winWin <- as.numeric(df$winWin)
df$winPlace <- as.numeric(df$winPlace)
df$refereeNationality <- as.numeric(df$refereeNationality)
df$assistant1Nationality <- as.numeric(df$assistant1Nationality)
df$assistant2Nationality <- as.numeric(df$assistant2Nationality)
df$homeOrAway <- as.numeric(df$homeOrAway)
df$result <- as.numeric(df$result)
df$host <- as.numeric(df$host)
df$isHostCountry <- as.numeric(df$isHostCountry)
#Set seed
set.seed(474)
#Set number of clusters
k <- 2
kmeansResult <- kmeans(df, k)

#Observe clusters
(kmeansResult$cluster)

plot <- fviz_cluster(kmeansResult, df)
plot

########################################################################
## Decision Trees
########################################################################
df <- dfOrig

# Set the random seed
set.seed(474)
# Split data set
trainRows <- sample(1:nrow(df),0.80*nrow(df))
train <- df[trainRows, ]
test <- df[-trainRows, ]

#create a function to display results of decision tree
get_results <- function(clf){
  #get predictions from the tree
  predictions <- predict(clf, test, type = "class")
  #visualize the tree
  fancyRpartPlot(clf)
  #Get cross validation table
  cv <- table(predictions, test$winWin)
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

formula <- formula(winWin~., data = train)
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
# Set the random seed
set.seed(474)
# Split data set
trainRows <- sample(1:nrow(df),0.80*nrow(df))
train <- df[trainRows, ]
test <- df[-trainRows, ]

NB_object<- naive_bayes(winWin~., data=train)
NB_prediction<-predict(NB_object, test[ , -which(names(test) %in% c("winWin"))], type = c("class"))
head(predict(NB_object, test, type = "class"))
table(NB_prediction,test$winWin)

########################################################################
## Random Forest
########################################################################

# Set the random seed
set.seed(474)
# Split data set
trainRows <- sample(1:nrow(df),0.80*nrow(df))
train <- df[trainRows, ]
test <- df[-trainRows, ]

n <- names(train)
f <- as.formula(paste("winWin ~", paste(n[!n %in% "winWin"], collapse = " + ")))

# Set up RF
rf <- randomForest(f, data = train)
print(rf)
# Make predictions
predictions <- predict(rf, test) 
cv = (table(predictions, test$winWin))
print(cv)
accuracy <- ((cv[1,1] + cv[2,2])/length(test$winWin))
print(accuracy)

########################################################################
## k-Nearest Neighbor
########################################################################
# copy df
dfOrig <- df

# get all numeric variables
df$year <- as.numeric(df$year)
df$time <- as.numeric(df$time)
df$stage <- as.numeric(df$stage)
df$winWin <- as.numeric(df$winWin)
df$winPlace <- as.numeric(df$winPlace)
df$refereeNationality <- as.numeric(df$refereeNationality)
df$assistant1Nationality <- as.numeric(df$assistant1Nationality)
df$assistant2Nationality <- as.numeric(df$assistant2Nationality)
df$homeOrAway <- as.numeric(df$homeOrAway)
df$result <- as.numeric(df$result)
df$host <- as.numeric(df$host)
df$isHostCountry <- as.numeric(df$isHostCountry)

# Set the random seed
set.seed(474)
# Split data set
trainRows <- sample(1:nrow(df),0.80*nrow(df))
train <- df[trainRows, ]
test <- df[-trainRows, ]


# Set k
k <- 4
# Fit the model
kNN_fit <- class::knn(train=train, test=test, cl=train$winWin, k = k, prob=TRUE)
print(kNN_fit)

## Check the classification accuracy
cv = (table(kNN_fit, test$winWin))
print(cv)
accuracy <- ((cv[1,1] + cv[2,2])/length(test$winWin))
print(accuracy)

#visualizaing KNN
plot(kNN_fit)
plot.df <- data.frame(test, predicted = kNN_fit)
plot.df1 = data.frame(x = plot.df$isHostCountry, 
                      y = plot.df$winWin, 
                      predicted = plot.df$predicted)

find_hull = function(df) df[chull(df$x, df$y), ]
boundary = ddply(plot.df1, .variables = "predicted", .fun = find_hull)

ggplot(plot.df, aes(isHostCountry, winWin, color = predicted, fill = predicted)) + 
  geom_point(size = 5) + 
  geom_polygon(data = boundary, aes(x,y), alpha = 0.5)


########################################################################
## Support Vector Machine
########################################################################
df <- dfOrig

# Set random seed
set.seed(474)
# Split data set
trainRows <- sample(1:nrow(df),0.80*nrow(df))
train <- df[trainRows, ]
test <- df[-trainRows, ]
# Set up SVM
clf <- svm(winWin~., data=train, kernel="radial", scale=FALSE)
print(clf)

# Get Predictions
(predictions <- predict(clf, test, type="class"))

# Get confusion matrix
(cv <- table(predictions, test$winWin))
print(cv)
accuracy <- ((cv[1,1] + cv[2,2])/length(test$winWin))
print(accuracy)

#Visualizing SVM
plot(clf, train, goalsFor ~ goalsAgainst)
