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


########################################################################
## Data Visualization
########################################################################

########################################################################
## Association Rules
########################################################################

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
