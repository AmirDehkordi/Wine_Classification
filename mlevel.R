# A
# Importing data
library(dplyr)
df <- read.csv('winedata.csv')
sum(duplicated(df))
df <- distinct(df)

# Exploratory Data Analysis
library(corrplot)
head(df)
summary(df)
dfcorr <- cor(df[,-c(12,13)])
corrplot(dfcorr, method = "circle")

# Computing the mean and standard deviation of the variables
library(dplyr)
submean <- aggregate(. ~ wine, data = df, FUN = mean)[,-13]
subsd <- aggregate(. ~ wine, data = df, FUN = sd)[,-13]
df_split <- split(df, df$wine)

# Conducting the t-test for variables in a for loop
out <- c()
for (i in 1:11){
  cat("t-test between wine and ", colnames(df[i]))
  a = t.test(df_split$Red[,i], df_split$White[,i], var.equal = TRUE)
  b = t.test(df_split$Red[,i], df_split$White[,i], var.equal = TRUE)$p.value
  print(a)
  out <- c(out, b)
}

# Creating the table of Means, STDs, and P-Values
out <- c(NA,out)
subdf <- rbind(submean, subsd, out)
rownames(subdf) <- c("Red Mean", "White Mean", "Red STD", "White STD", "P-Value")
round(subdf[,-1],4)



# B
# Splitting data into train and test data
set.seed(23)
training <- sample(c(TRUE, FALSE), size = nrow(df), replace = TRUE, prob = c(0.8, 0.2))
df.train <- df[training,]
df.test <- df[!training,]

# If the color is white it is 1, otherwise 0
df.train$wine[df.train$wine == "White"] <- as.numeric(1)
df.test$wine[df.test$wine == "White"] <- as.numeric(1)
df.train$wine[df.train$wine == "Red"] <- as.numeric(0)
df.test$wine[df.test$wine == "Red"] <- as.numeric(0)
df.train$wine <- as.factor(df.train$wine)
df.test$wine <- as.factor(df.test$wine)

# Creating plots for the training and test data
par(mfrow=c(1,2))
gptrain <- table(df.train$wine)
barplot(gptrain, xlab = "Color of Wine", ylab = "Frequency", main = "Frequency in Training set",
        col = c("#eb8060", "#a1e9f0"), names = c("Red", "White"))

gptest <- table(df.test$wine)
barplot(gptest, xlab = "Color of Wine", ylab = "Frequency", main = "Frequency in Test set",
        col =  c("#eb8060", "#a1e9f0"), names = c("Red", "White"))

gptrain.prop <- prop.table(gptrain)
barplot(gptrain.prop, xlab = "Color of Wine", ylab = "Percentage", main = "Percentage in Training set",
        col =  c("#eb8060", "#a1e9f0"), names = c("Red", "White"))

gptest.prop <- prop.table(gptest)
barplot(gptest.prop, xlab = "Color of Wine", ylab = "Percentage", main = "Percentage in Test set",
        col =  c("#eb8060", "#a1e9f0"), names = c("Red", "White"))
par(mfrow=c(1,1))

gpdf <- rbind(gptrain, prop.table(gptrain), gptest, prop.table(gptest))
gptrain
round(gpdf,4)

# C
# Logistic Regression
logistic.model <- glm(wine ~ . - quality, data = df.train, family="binomial")
summary(logistic.model)

# Test the model on the test data
prob <- predict(logistic.model, df.test, type = "response")
pred.test.glm <- ifelse(prob>0.5, 1, 0)
logistic.error <- mean(df.test$wine != pred.test.glm)
logistic.error


# Linear Discriminant Analysis
# Fit a model to the training data
library(MASS)
lda.model <- lda(wine ~ . - quality, data = df.train)
lda.model

# Test the model on the test data
pred.test.lda <- predict(lda.model, df.test)$class
lda.error <- mean(df.test$wine != pred.test.lda)
lda.error


# Decision Tree Classifier
library(tree)
tree.model <- tree(wine ~ . - quality, data = df.train)
summary(tree.model)

# Prune the tree using cross validation
tree.cv.model <- cv.tree(tree.model, FUN = prune.misclass, K = 10)
tree.cv.model

plot(tree.cv.model$size, tree.cv.model$dev, type='b')

tree.prune.model <- prune.tree(tree.model, best = 9)
summary(tree.prune.model)

# Plot the pruned tree
plot(tree.prune.model)
text(tree.prune.model, pretty = 0)

# Test the model on the test data
pred.test.tree <- predict(tree.prune.model, df.test, type = "class")
tree.error <- mean(pred.test.tree != df.test$wine)
tree.error


# Bagging
library(rpart)
library(ipred)
bag.model <- bagging(wine ~ . - quality, data = df.train, coob = F , nbagg = 100)
bag.model

# Find important variables
library(caret)
vi <- data.frame(var=names(df.train[,-13]), imp=varImp(bag.model))
vi
vi_plot <- vi[order(vi$Overall, decreasing=TRUE),][0:3,]
barplot(vi_plot$Overall, names.arg=rownames(vi_plot), horiz=F, col='steelblue', ylab='Variable Importance')

# Test the model on the test data
pred.test.bag <- predict(bag.model, df.test, type = "class")
bag.error <- mean(pred.test.bag != df.test$wine)
bag.error


# Random Forest
library(randomForest)
rf.model <- randomForest(wine ~ . - quality, data = df.train, ntree = 100)
rf.model

# Tune the hyperparameter of 'mtry'
mtry <- tuneRF(df.train[-12:-13], df.train$wine, ntreeTry = 1000,
               stepFactor = 1.5,improve = 0.01, trace = TRUE, plot = TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
mtry
best.m

# Build the Random Forest based on the optimal 'mtry'
rf.model.best <- randomForest(wine ~ . - quality, data = df.train, importance = TRUE, mtry = best.m, ntree = 1000)
rf.model.best

# Find important variables
importance(rf.model.best)
data <- data.frame(importance(rf.model.best)[,4])
barplot(sort(data[,1], decreasing = T)[1:3], horiz=F, col='steelblue', ylab='Variable Importance',
        main="Variable Importance for Random Forest", names=c("total.sulfur.dioxide", "chlorides", "volatile.acidity"))

# Test the model on the test data
pred.test.rf <- predict(rf.model.best, df.test, type = "class")
rf.error <- mean(pred.test.rf != df.test$wine)
rf.error


# Check the important variables in Decision Tree, Bagging, and Random Forest
library(ggplot2)
library(GGally)
ggpairs(df[,c(2,5,7,13)], progress = F)


nerr <- cbind(c("Logistic Regression", "Linear Discriminant Analysis", 
                "Decision Tree", "Bagging", "Random Forest"))
err <- cbind(c(logistic.error, lda.error, tree.error, bag.error, rf.error))
err <- round(err * 100, 4)
errors <- cbind(nerr,err)
errors

