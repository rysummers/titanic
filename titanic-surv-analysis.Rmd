---
title: An R Markdown document converted from "/Users/ryan_summers/Downloads/titanic-survival-prob-analysis.ipynb"
output: html_document
---

**Load Packages**

```{r}
# load Tidyverse for packages readr and dplyr
library (tidyverse)
library(dplyr)

# load R Neural Network package
library(neuralnet)

# visualizations
library(ggplot2)
library(ggcorrplot)
library(corrplot)
library(viridis)
library(hrbrthemes)
library(ggthemes)
```

**Load in the data**

```{r}
# df for training set
t.train <- read_csv('/Users/ryan_summers/Documents/Kaggle/titanic_comp/train.csv')
head(t.train)
```

```{r}
#df for test set
t.test <- read_csv('/Users/ryan_summers/Documents/Kaggle/titanic_comp/test.csv')
head(t.test)
```

**View our data structures**

```{r}
str(t.train)
```

```{r}
str(t.test)
```

**Survived:** Survival Indicator (0 = No, 1 = Yes)
**Name:** Passenger name
**Gender:** Passenger’s gender
**GenderNum:** Passenger’s numeric gender (0 = Female, 1 = Male)
**Age:** Age in years
**SiblingSpouse:** Number of passengers on ship who are this person’s brother, sister, or spouse
**ParentChild:** Number of passengers on ship who are this person’s parent or child
**PClass:** Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
**Fare:** Passenger fare
**Embarked:** location passenger embarked from (C = Cherbourg, Q = Queenstown, S = Southampton)

**Combine Datasets and Add New Column to Distinguish between Train and Test** 

We are combining both train and test data sets to make it easier to clean the data at once.  Later we will seperate the datasets to train the models for predictions.

```{r}
# add a boolean column with True for the train dataset and False for test dataset
t.train$train <- TRUE
t.test$train <- FALSE

# combine datasets with bind_rows() function
t.full <- bind_rows(t.train, t.test)

# confirm 418 NAs in Survived column for rows from t.test(418). See glimpse(t.test) output above.
t.full %>% 
    filter(is.na(Survived)) %>% 
    nrow()
```

**Create Dummy Variables(factors)**

Predictor variables Sex, Pclass, Embarked, and response variable Survied needs to be factored.

```{r}
t.full$Sex <- as.factor(t.full$Sex)
t.full$Pclass <- as.factor(t.full$Pclass)
t.full$Survived <- as.factor(t.full$Survived)
t.full$Embarked <- as.factor(t.full$Embarked)
```

```{r}
glimpse(t.full)
```

**Visualizing the Data**

```{r}
ggplot(t.full, aes(x=as.factor(Pclass), fill=as.factor(Pclass) )) + 
    ggtitle("Number of Passengers per Class") + 
    geom_bar()  + 
    scale_fill_discrete(name = 'Class', labels = c('1st', '2nd', '3rd')) +
    theme_economist()
```

```{r}
ggplot(t.full, aes(x=Fare, fill=..x..)) + 
    geom_histogram(color='black', alpha=0.7, bins=30) + 
    scale_fill_gradient(low='blue', high='yellow')
```

```{r}
# create a correlation matrix and label correlation variable 't.corr'
t.corr <- cor(t.full[,c("Age", "Fare","Parch","PassengerId","SibSp")], use="complete")
corrplot(t.corr, addCoef.col = 'grey45')
```
No strong correlations exist


**Finding Missing Values**

```{r}
# use summary() to find an NA's
summary(t.full)
```

```{r}
# library needed for NA visualization using aggr function
library(VIM)

# We are going to use the aggr function for this purpose
# The plot on the left shows the percentage of missing values for each variable
# The plot on the right shows the combination of missing values between variables
summary(aggr(t.full))
```

Based on the results above, NA's are located in variables Survived, Age, and Fare.  Remember NA's in Survied is for our prediction so we can ignore those for now. Pay attention to Cabin and Embarked. No NA's are listed but looking at the dataset we can there is missing datapoints

```{r}
subset(t.full, is.na(t.full$Embarked))
```

As you can see, these two ladies are in the same Pclass, Cabin, and have same Ticket number. Let's see if we can discover some trends based on these characteristics.

```{r}
# create dataframe without the two passengers above
t.embark <- t.full %>% 
    filter(PassengerId != 62 & PassengerId != 830)

# count passengers by class and embarked
t.embark %>% 
    group_by(Embarked, Pclass) %>% 
    count()
```

```{r}
ggplot(t.embark, aes(x=Embarked, fill=Pclass)) + 
    geom_bar() + 
    scale_fill_viridis(discrete = T)  +
    ggtitle('# of Passengers by Embarked Location') + 
    theme(plot.title = element_text(size = 18, face = "bold"))

# visualization of proportion of class to each location
colors <- c("#00405b", "#008dca", "#c0beb8", "#d70000", "#7d0000")
colors <- setNames(colors, levels(t.embark$Pclass))
values = c("1st Class" = "#7D0000", "2nd Class" = "#D70000","3rd Class" = "#C0BEB8")

tbl <- xtabs(~ Embarked + Pclass, t.full)
proptbl <- proportions(tbl, margin = "Embarked")
proptbl <- as.data.frame(proptbl)
proptbl <- proptbl[proptbl$Freq != 0, ]

ggplot(proptbl, aes(Embarked, Freq, fill = Pclass)) +
  geom_col(position = position_fill()) +
  geom_text(aes(label = scales::percent(Freq)),
            colour = "white",
            position = position_fill(vjust = .5)) +
  scale_fill_manual(values = colors) +
  guides(fill = guide_legend(title = "Passenger Class")) + 
    ggtitle('Class Makeup by Embarked Location') + 
    theme(plot.title = element_text(size = 18, face = "bold")) 
```

```{r}
# boxplot to compare fare prices to passenger class grouped by embarked location with $80 fare paid by passengers shown on the y-intercept
ggplot(t.embark, aes(Embarked, Fare, fill = Pclass)) +
    geom_boxplot() +
    geom_hline(aes(yintercept = 80), linetype = "dashed")
```

By comparing the interquartile ranges, we can deduce that the passengars were more probable to have embarked from Cherbourg (c) based on the median fare price and being in first class.

```{r}
# update Embarked values for passengers
t.full[c(62,830), 'Embarked']  <- 'C'

# check if update worked
t.full %>% 
    filter(PassengerId %in% c(62, 830))
```

Next lets look at how Age NAs are spread using boxplots

```{r}
boxplot(t.full$Age)
```

```{r}
# distribution of NaN for Age group by Embarked and Pclass
t.nan <- t.full %>% 
    filter(is.na(Age))
t.nan %>% 
    group_by(Embarked, Pclass) %>% 
    count()
```

```{r}
# geom_box() plot automatically removes NA values
ggplot(t.full, aes(Pclass, Age)) + geom_boxplot()
ggplot(t.full, aes(Embarked, Age)) + geom_boxplot()
ggplot(t.full, aes(Pclass, Age, fill = Embarked)) + geom_boxplot()
```


```{r}
# calculate percentage of NAs in Age variable
percent(sum(is.na(t.full$Age))/boxplot.stats(t.full$Age)$n[1], by=0.1)
```

```{r}
# Age boxplot stats
boxplot.stats(t.full$Age)
```

**stats:** a vector of length 5, containing the extreme of the lower whisker, the lower ‘hinge’, the median, the upper ‘hinge’ and the extreme of the upper whisker. (extreme should not to be confused with outliers which are outside the whiskers)

**n:** the number of non-NA observations in the sample.

**conf:** the lower and upper extremes of the ‘notch'. The notches extend to +/-1.58 IQR/sqrt(n).

**out:** the values of any data points which lie beyond the extremes of the whiskers.

Let's build a regression model to predict the remaining missing ages. Like the mean, OLS models are susceptible to outliers. Therefore, we will use the vector range calculated above for **$stats**. 

```{r}
# new dataframe with NAs removed
rem.na <- t.full %>% drop_na(Age)

# create dataframe by setting Max Age being less than or equal to extreme upper whisker 66 from above
t.age <- rem.na$Age <= 66

# check to see if new Max is 66
summary(rem.na[t.age, ]$Age)
```

**Create Regression**

We will not use ticket number or cabin due to cabin having many missing values and ticket number likely not adding any value in predicting the age of passengers.


```{r}
# linear equation
regr.eq <- 'Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked'

# linear model with removed NA df
reg.model <- lm(formula = regr.eq, data = rem.na[t.age,])

# check regression model for statistical significance
summary(reg.model)
```

With a p-value of the F-statistic being <.05 we can interpret this model. With a Multiple R of .24, 24% of the variations in Age were explained by changes in the predictor variables. 

For the coefficients, Parch and Fare are insignificant, with Embarked Q & S having weak evidence against the null.

> *the null is that the coefficients are equal to 0 and have no statistical relationship with the response (Age)*

For the sake of this prediction, I will only remove Parch and Fare from the model.

```{r}
# linear equation
regr.eq2 <- 'Age ~ Pclass + Sex + SibSp + Embarked'

# linear model
reg.model2 <- lm(formula = regr.eq2, data = rem.na[t.age,])

# check regression model for statistical significance
summary(reg.model2)
```

**Age Prediction**

With the model created, we need to filter original dataframe Age data by NA to make the predictions.

```{r}
# filter for Age by NA
age.na <- t.full[is.na(t.full$Age), c("Pclass", "Sex", "SibSp", "Embarked")]

# store prediction to a new variable
pred.age.res <- predict(reg.model2, newdata = age.na)

# assign new predictions to NaN Age values
t.full[is.na(t.full$Age), "Age"] <- pred.age.res

# check new IQR, Range and Mean stats
summary(t.full$Age)                                  
```

NaN values from Age are gone but we have negative values as noted by the Min. This is likely due to a linear violation of OLS assumptions. However, within the scope of this project, it's not worth running assumption tests on the predictive values for Age. We will replace zero and negative values with the min positive value of Age.

```{r}
# filter Age data that is greater than zero to determine the minimum value
min_age <- t.full %>%
    filter(Age > 0) %>%
    summarize(min = min(Age))

# replace negative and zero values for age with (0.17)
t.full[t.full$Age <= 0, "Age"] <- min_age

# check Age stats
summary(t.full$Age)
```

On to Fare NaN values

```{r}
boxplot(t.full$Fare)
summary(t.full$Fare)
```

```{r}
subset(t.full, is.na(t.full$Fare))
```

Only one entry is noted. Let's see if we can find similar tickets

```{r}
# load stringr for searching regex patterns
library(stringr)

# filter based on partial matches to find like string patterns
t.full %>% 
  filter(str_detect(tolower(Ticket), pattern = "3701"))
```

Based on the above, Ticket#s 370129 all originiated from location S and are in third class. Thus, we can deduce that this was likely an entry error and has a Fare of 20.2125

```{r}
# update Passenger 1044 Ticket and Fare
t.full[t.full$PassengerId == 1044, 'Ticket']  <- '370129'
t.full[t.full$PassengerId ==1044, 'Fare'] <- 20.2125

# check
t.full %>% filter(Ticket == '370129')

t.full[t.full$Ticket == '370129',]
```

Finally lets look at Cabin NAs

```{r}
prop <- t.full %>% 
    filter(is.na(Cabin)) %>% 
    group_by(Pclass) %>% 
    count()

prop['%'] = prop['n']/colSums(prop['n'])
prop
```

As we can see, there is a high proportion of passengers with missing values for Cabin in third class (68.3%). This could be an indication of passengers who did not return from the voyage.  We will keep things simple and fill all Cabin NAs with 'Missing' and then extract the Cabin class number to the first alphabetical string value.

```{r}
# replace NAs with 'Missing'
t.full$Cabin <- t.full$Cabin %>% replace_na('Missing')

# check for NAs
t.full %>% filter(is.na(Cabin)) %>% count()
```

```{r}
# extract first character from Cabin and update the variable in the df
t.full <- t.full %>% 
    mutate(Cabin = substr(Cabin,1,1))

# factor variable Cabin
t.full$Cabin <- as.factor(t.full$Cabin)

# check
head(t.full$Cabin)
```

# Feature Engineering

The feature engineering pipeline is the preprocessing steps that transform raw data into features that can be used in machine learning algorithms, such as predictive models. Predictive models consist of an outcome variable and predictor variables, and it is during the feature engineering process that the most useful predictor variables are created and selected for the predictive model. Automated feature engineering has been available in some machine learning software since 2016. Feature engineering in ML consists of four main steps: Feature Creation, Transformations, Feature Extraction, and Feature Selection.
Feature engineering consists of creation, transformation, extraction, and selection of features, also known as variables, that are most conducive to creating an accurate ML algorithm. These processes entail:

**Feature Creation:** Creating features involves identifying the variables that will be most useful in the predictive model. This is a subjective process that requires human intervention and creativity. Existing features are mixed via addition, subtraction, multiplication, and ratio to create new derived features that have greater predictive power.  

**Transformations:** Transformation involves manipulating the predictor variables to improve model performance; e.g. ensuring the model is flexible in the variety of data it can ingest; ensuring variables are on the same scale, making the model easier to understand; improving accuracy; and avoiding computational errors by ensuring all features are within an acceptable range for the model. 

**Feature Extraction:** Feature extraction is the automatic creation of new variables by extracting them from raw data. The purpose of this step is to automatically reduce the volume of data into a more manageable set for modeling. Some feature extraction methods include cluster analysis, text analytics, edge detection algorithms, and principal components analysis.

**Feature Selection:** Feature selection algorithms essentially analyze, judge, and rank various features to determine which features are irrelevant and should be removed, which features are redundant and should be removed, and which features are most useful for the model and should be prioritized.

**Family or Alone?**

We will analyze variables SibSp and Parch to determine family size and if passengers were alone to determine if it had any impact on survivability.

* SibSp: number of siblings / spouses aboard the ship
* Parch: numbuer of parents / children aboard

```{r}
# add variables SibSp and Parch to get family size
t.full$Alone <- t.full['SibSp'] + t.full['Parch']

# use an if statement to apply 1 if alone and 0 if not
t.full$Alone <- ifelse(t.full$Alone>0,0,1)

# Make Alone variable a factor
t.full$Alone <- as.factor(t.full$Alone)

# check
head(t.full)
```

```{r}
t.full %>% 
    filter(!is.na(Survived)) %>%
    group_by(Alone) %>%
    count(Survived)
```

```{r}
#ggplot(t.full, aes(x=Alone)) + 
#geom_bar(y=Survived) 
t.full %>% 
    filter(!is.na(Survived)) %>%
    filter(Alone == 1) %>%
    ggplot(aes(x=Survived)) +
    geom_bar(fill= c('#CC6666', '#66CC99')) +
    theme(plot.title = element_text(size=22)) +
    labs(title='Alone')

t.full %>% 
    filter(!is.na(Survived)) %>%
    filter(Alone == 0) %>%
    ggplot(aes(x=Survived)) +
    geom_bar(fill= c('#CC6666', '#66CC99')) +
    theme(plot.title = element_text(size=22)) +
    labs(title='With Family')
```

**Title?**

Let's seperate honorific titles to see if they add any value

```{r}
# extract titles from Name column
t.full <- t.full %>% 
    mutate(Honorific = str_extract(Name, "([A-Za-z]+\\.)"))

# check all titles
unique(t.full$Honorific)
```

Lets consolidate honorific titles:

Capt. & Major.  = Officer.
Lady., Countess., Don., Sir., Jonkheer. & Dona. = Royal.
Mlle. & Ms. = Miss.
Mme = Mrs.

```{r}
t.full$Honorific <- ifelse(t.full$Honorific %in% c('Capt.','Major.','Col.'), 'Military', t.full$Honorific)
t.full$Honorific <- ifelse(t.full$Honorific %in% c('Lady.','Countess.','Don.','Sir.','Jonkheer.','Dona.'), 'Royal', t.full$Honorific)
t.full$Honorific <- ifelse(t.full$Honorific %in% c('Mlle.','Ms.'), 'Miss.', t.full$Honorific)
t.full$Honorific <- ifelse(t.full$Honorific %in% c('Mme.'), 'Mrs.', t.full$Honorific)
```

```{r}
# factor the Honorific variable
t.full$Honorific <- as.factor(t.full$Honorific)
```

```{r}
t.full %>%
    group_by(Honorific) %>%
    count()

t.full %>% 
    filter(!is.na(Survived)) %>%
    group_by(Honorific) %>%
    count(Survived)
```

# Building the Model

We will use four different models to make our predictions.
1. Neural Network
2. Logit
3. Decision Tree
4. Random Forest

```{r}
# create the equation for the model
surveq <- as.formula('Survived ~ Pclass + Sex + Age + SibSp + Parch + 
    Fare + Cabin + Embarked + Alone + Honorific') 

# create new df with cleaned trained data - training data should have 891 rows
t.trained <- t.full %>%
    filter(train == TRUE) %>%
    select(-c(Name, Ticket, train)) # easier to select columns not wanted

# confirmed desired variables are clean and transformed
str(t.trained)
```

```{r}
# create new df with cleaned test data - test data should have 418 rows
t.test <- t.full %>%
    filter(train == FALSE) %>%
    select(-c(Survived, Name, Ticket, train)) # easier to select columns not wanted

# confirmed desired variables are clean and transformed
str(t.test)
```

```{r}
# first for trained data
t.trained <- t.trained %>%
    # create dummy variables with binary outcomes
    mutate(Male = as.integer(ifelse(Sex == 'male', 1, 0)),
           Female = as.integer(ifelse(Sex == 'female', 1, 0)),
           Cherbourg = as.integer(ifelse(Embarked == 'C', 1, 0)),
           Queenstown = as.integer(ifelse(Embarked == 'Q', 1, 0)),
           Southampton = as.integer(ifelse(Embarked == 'S', 1, 0)),
           FirstClass = as.integer(ifelse(Pclass == '1', 1, 0)),
           SecondClass = as.integer(ifelse(Pclass == '2', 1, 0)),
           ThirdClass = as.integer(ifelse(Pclass == '3', 1, 0)),
           Mr. = as.integer(ifelse(Honorific == 'Mr.', 1, 0)),
           Mrs. = as.integer(ifelse(Honorific == 'Mrs.', 1, 0)),
           Miss. = as.integer(ifelse(Honorific == 'Miss.', 1, 0)),
           Master. = as.integer(ifelse(Honorific == 'Master.', 1, 0)),
           Rev. = as.integer(ifelse(Honorific == 'Rev.', 1, 0)),
           Dr. = as.integer(ifelse(Honorific == 'Dr.', 1, 0)),
           Military = as.integer(ifelse(Honorific == 'Military', 1, 0)),
           M = as.integer(ifelse(Cabin == 'M', 1, 0)),
           C = as.integer(ifelse(Cabin == 'C', 1, 0)),
           E = as.integer(ifelse(Cabin == 'E', 1, 0)),
           G = as.integer(ifelse(Cabin == 'G', 1, 0)),
           D = as.integer(ifelse(Cabin == 'D', 1, 0)),
           A = as.integer(ifelse(Cabin == 'A', 1, 0)),
           B = as.integer(ifelse(Cabin == 'B', 1, 0)),
           F = as.integer(ifelse(Cabin == 'F', 1, 0)),
           T = as.integer(ifelse(Cabin == 'T', 1, 0))) %>%
    # select desired columns
    select(Survived, Alone, Age, SibSp, Parch, Fare, Male, Female, Cherbourg, Queenstown, Southampton, FirstClass, SecondClass, ThirdClass, Mr., Mrs., Miss., Master., Rev., Dr., Military, M, C, E, G, D, A, B, F, T)
```

```{r}
# for test data
t.test <- t.test %>%
    # create dummy variables with binary outcomes
    mutate(Male = as.integer(ifelse(Sex == 'male', 1, 0)),
           Female = as.integer(ifelse(Sex == 'female', 1, 0)),
           Cherbourg = as.integer(ifelse(Embarked == 'C', 1, 0)),
           Queenstown = as.integer(ifelse(Embarked == 'Q', 1, 0)),
           Southampton = as.integer(ifelse(Embarked == 'S', 1, 0)),
           FirstClass = as.integer(ifelse(Pclass == '1', 1, 0)),
           SecondClass = as.integer(ifelse(Pclass == '2', 1, 0)),
           ThirdClass = as.integer(ifelse(Pclass == '3', 1, 0)),
           Mr. = as.integer(ifelse(Honorific == 'Mr.', 1, 0)),
           Mrs. = as.integer(ifelse(Honorific == 'Mrs.', 1, 0)),
           Miss. = as.integer(ifelse(Honorific == 'Miss.', 1, 0)),
           Master. = as.integer(ifelse(Honorific == 'Master.', 1, 0)),
           Rev. = as.integer(ifelse(Honorific == 'Rev.', 1, 0)),
           Dr. = as.integer(ifelse(Honorific == 'Dr.', 1, 0)),
           Military = as.integer(ifelse(Honorific == 'Military', 1, 0)),
           M = as.integer(ifelse(Cabin == 'M', 1, 0)),
           C = as.integer(ifelse(Cabin == 'C', 1, 0)),
           E = as.integer(ifelse(Cabin == 'E', 1, 0)),
           G = as.integer(ifelse(Cabin == 'G', 1, 0)),
           D = as.integer(ifelse(Cabin == 'D', 1, 0)),
           A = as.integer(ifelse(Cabin == 'A', 1, 0)),
           B = as.integer(ifelse(Cabin == 'B', 1, 0)),
           F = as.integer(ifelse(Cabin == 'F', 1, 0)),
           T = as.integer(ifelse(Cabin == 'T', 1, 0))) %>%
        # select desired columns
    select(PassengerId, Age, Alone, SibSp, Parch, Fare, Male, Female, Cherbourg, Queenstown, Southampton, FirstClass, SecondClass, ThirdClass, Mr., Mrs., Miss., Master., Rev., Dr., Military, M, C, E, G, D, A, B, F, T)
```

**Neural Network**

```{r}
# create new formula with new variable names above
nnet.eq <- as.formula('Survived ~ Age + SibSp + Parch + Fare + Male + Female + Cherbourg + Queenstown + Southampton + FirstClass + SecondClass + ThirdClass + Mr. + Mrs. + Miss. + Master. + Rev. + Dr. + Military + M + C + E + G + D + A + B + F + T')
set.seed(8)
```

```{r}
# load library for train function
library(caret)
```

```{r}
set.seed(1, sample.kind = "Rounding")
control <- trainControl(method = "cv", number = 10, p = .9)
tuning <- data.frame(size = seq(100), decay = seq(.01,1,.1))

# Predicting survival by using a neural network
set.seed(1, sample.kind = "Rounding")

#  suppressWarnings() and invisible(capture.output()) suppresses the iterations output as it is excessive
# I upped the MaxNWts to accommodate the size of my model due to many of the variables being factored
suppressWarnings(invisible(capture.output(t.nnet <- train(nnet.eq, data=t.trained,
                  method = "nnet",
                  tuneGrid = tuning,
                  trControl = control))))
```

```{r}
t.nnet$bestTune
```

```{r}
plot(t.nnet)
```

```{r}
# make predictions
nn.pred <- predict(t.nnet, t.test)

nn.result <- data.frame(PassengerID = t.test$PassengerId, Survived = nn.pred)

# write csv for submission
write.csv(nn.result, file = 'nn.titanic.preds.csv', row.names = FALSE)
```

```{r}
head(nn.result)
```

```{r}
# all predictors stored in pred variable
pred <- c('Alone', 'Age', 'SibSp', 'Parch', 'Fare', 'Male', 'Female', 'ThirdClass', 'SecondClass', 'FirstClass', 'Southampton', 'Queenstown', 'Cherbourg', 'Mr.','Mrs.','Miss.','Master.','Rev.','Dr.','Military','M','C','E','G','D','A','B','F','T')
```

**Logit Regression**

```{r}
log_reg <- glm(nnet.eq, data = t.trained, family = "binomial")
```

```{r}
# predict probability of survival with logistic regression model
log.reg.test <- t.test %>%
    mutate(prob = predict(log_reg, t.test, type = "response"),
            Survived = ifelse(prob > 0.50, 1, 0))

# check predictions
head(log.reg.test)

# create csv file for submission
logit.result <- data.frame(PassengerID = log.reg.test$PassengerId, Survived = log.reg.test$Survived)

# write csv for submission
write.csv(logit.result, file = 'logit.titanic.preds.csv', row.names = FALSE)
```

**Decision Tree**

```{r}
library(rpart)

# build classification tree model
decTree <- rpart(nnet.eq, data = t.trained, 
                 method = "class")
```

```{r}
# make predictions with decision tree model
dec.tree.test <- t.test %>%
    mutate(Survived = predict(decTree, t.test, type = "class"))

# check Survived predictions
head(dec.tree.test)

# create csv file for submission
dtree.result <- data.frame(PassengerID = dec.tree.test$PassengerId, Survived = dec.tree.test$Survived)

# write csv for submission
write.csv(dtree.result, file = 'dtree.titanic.preds.csv', row.names = FALSE)
```

**Random Forest**

```{r}
# load randomForest library
library(randomForest)
```

For our random forest model, we'll keep the ntree (number of trees to grow) at 500 which is common practice. For mtry (number of variables randomly sampled as candidates at each split) we have to set it as the square root of the number of independent/predictor variables for classification problems. Where p is the number of our dependent/predictor variables, this gives sqrt(30) = 5.48 round up to 6. (see below)

```{r}
sqrt(length(t.trained))
```

```{r}
# make forest model
rand_for <- randomForest(nnet.eq, data = t.trained,
                         ntree = 500, mtry = 6)
```

```{r}
# make predictions with Random Forest model
rand.for.test <- t.test %>%
    mutate(Survived = predict(rand_for, t.test))

# check Survived predictions
head(rand.for.test)

# create csv file for submission
forest.result <- data.frame(PassengerID = rand.for.test$PassengerId, Survived = rand.for.test$Survived)

# write csv for submission
write.csv(forest.result, file = 'foreset.titanic.preds.csv', row.names = FALSE)
```

