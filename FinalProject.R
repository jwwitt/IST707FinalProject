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

# Set working directory
setwd("~/Desktop/IST 707/IST707FinalProject")

# Load in the data
worldCupData <- read.csv("WorldCups.csv")
matchData <- read.csv("WorldCupMatches.csv")


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
                         "referee", "assistant1", "assistant2")

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

# get correct data types
df$year <- as.factor(df$year)
df$time <- as.factor(df$year)
df$referee <- as.factor(df$referee)
df$assistant1 <- as.factor(df$assistant1)
df$assistant2 <- as.factor(df$assistant2)
df$refereeNationality <- as.factor(df$refereeNationality)
df$assistant1Nationality <- as.factor(df$assistant1Nationality)
df$assistant2Nationality <- as.factor(df$assistant2Nationality)
df$homeOrAway <- as.factor(df$homeOrAway)
df$result <- as.factor(df$result)

########################################################################
## Data Visualization
########################################################################

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

########################################################################
## Cosine Similarity
########################################################################

########################################################################
## Decision Trees
########################################################################

########################################################################
## Naive Bayes
########################################################################

########################################################################
## Random Forest
########################################################################

########################################################################
## k-Nearest Neighbor
########################################################################

########################################################################
## Support Vector Machine
########################################################################