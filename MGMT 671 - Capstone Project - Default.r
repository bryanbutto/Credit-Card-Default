install.packages("ISLR")

library("ISLR")

Default
?Default

getwd()

write.csv(Default, "Default.csv")

head(Default)

str(Default)

summary(Default)
# 333 out of 10,000 defaulted on their credit card
# 9,667 out of 10,000 did not default on their credit card

install.packages("pastecs")

library(pastecs)

round(stat.desc(Default[,3:4]),2)

pairs(Default)

Default$default1 <- ifelse(Default$default == "Yes", 1, 0)
Default$student1 <- ifelse(Default$student == "Yes", 1, 0)

str(Default)

round(cor(Default[,3:6]),2)

library(corrplot)

cors <- cor(Default[,3:6])
corrplot(cors)
corrplot(cors, method = "number")
# Almost no correlation between default and income and default and student.
# There is a strong negative correlation between student and income. (Students tend to have a lower income)

library(car)

scatterplot(balance ~ income | default, data=Default, regLine=FALSE, smooth=FALSE)

library(caret)

featurePlot(x=Default[,3:4], y=Default$default, type = c("g", "smooth"))
featurePlot(x=Default[,3:4], y=Default$default, type = c("g", "p", "smooth"))

library(ggplot2)

ggplot(Default, aes(x=default1)) + geom_histogram(alpha=2) + ggtitle("Default")
ggplot(Default, aes(x=student1)) + geom_histogram(alpha=2) + ggtitle("Student")
ggplot(Default, aes(x=balance)) + geom_histogram(alpha=2) + ggtitle("Balance")
ggplot(Default, aes(x=income)) + geom_histogram(alpha=2) + ggtitle("Income")
ggplot(Default, aes(x=default1)) + geom_density() + ggtitle("Default")
ggplot(Default, aes(x=student1)) + geom_density() + ggtitle("Student")
ggplot(Default, aes(x=balance)) + geom_density() + ggtitle("Balance")
ggplot(Default, aes(x=income)) + geom_density() + ggtitle("Income")

ggplot(Default, aes(default1)) + geom_boxplot() + ggtitle("Default")
ggplot(Default, aes(student1)) + geom_boxplot() + ggtitle("Student")
ggplot(Default, aes(balance)) + geom_boxplot() + ggtitle("Balance")
ggplot(Default, aes(income)) + geom_boxplot() + ggtitle("Income")

ggplot(Default, aes(x=default1, y=student1)) + geom_point() + ggtitle("Default vs. Student")
ggplot(Default, aes(x=default1, y=balance)) + geom_point() + ggtitle("Default vs. Balance")
ggplot(Default, aes(x=default1, y=income)) + geom_point() + ggtitle("Default vs. Income")

ggplot(Default, aes(x=default1, color=student)) + geom_density() + ggtitle("Default by Student")
ggplot(Default, aes(x=balance, color=student)) + geom_density() + ggtitle("Balance by Student")
ggplot(Default, aes(x=income, color=student)) + geom_density() + ggtitle("Income by Student")
ggplot(Default, aes(x=balance, color=default)) + geom_density() + ggtitle("Balance by Default")
ggplot(Default, aes(x=income, color=default)) + geom_density() + ggtitle("Income by Default")
# default appears to be more likely with a higher balance

set.seed(582)
Default.train <- createDataPartition(Default$default, p=3/4, list = FALSE)
head(Default.train, 10)
tail(Default.train, 10)

trainingset <- Default[Default.train,]
testingset <- Default[-Default.train, ]
head(trainingset)
head(testingset)
str(trainingset)
str(testingset)
summary(trainingset)
summary(testingset)

repeatedsplits <- createDataPartition(trainingset$default,p=.8, times = 10)
str(repeatedsplits)

repeatedsplits1 <- createFolds(trainingset$default, k=10, returnTrain=TRUE)
str(repeatedsplits1)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)

library(MASS)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.LDA <- train(default ~ student + balance + income, data=trainingset,
                    method="lda",
                    preProc = c("center", "scale"), 
                    trControl = controlobject)
Default.LDA
summary(Default.LDA)

confusionMatrix(Default.LDA)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.Log <- train(default ~ student + balance + income, data=trainingset,
                    method="glm",
                    family=binomial,
                    preProc = c("center", "scale"), 
                    trControl = controlobject)
Default.Log
summary(Default.Log)

confusionMatrix(Default.Log)

plot(Default.Log.Predict)

Default.Log.Predict.prob <- ifelse(Default.Log.Predict < 0.5, "No", "Yes")
head(Default.Log.Predict.prob,5)

table(Default.Log.Predict.prob, trainingset$default)

mean(Default.Log.Predict.prob == trainingset$default)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.PLS <- train(default ~ student + balance + income, data=trainingset,
                    method="pls",
                    trControl = controlobject)
Default.PLS

confusionMatrix(Default.PLS)
plot(Default.PLS)

install.packages("glmnet")

library(caret)
library(glmnet)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.PLR <- train(default ~ student + balance + income, data=trainingset, method="glmnet",
                    preProc = c("center", "scale"), 
                    trControl = controlobject)
summary(Default.PLR)
Default.PLR

confusionMatrix(Default.PLR)

plot(Default.PLR)

## Added tuning parameters
controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1) , .lambda = seq(.001, .2, length = 40))
set.seed(590)
Default.PLR <- train(default ~ student + balance + income, data=trainingset, method="glmnet",
                    preProc = c("center", "scale"),
                    tuneGrid = glmnGrid,
                    trControl = controlobject)
summary(Default.PLR)
Default.PLR

plot(Default.PLR)

install.packages("pamr")

library(pamr)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.PAM <- train(default ~ student + balance + income, data=trainingset,
                    method="pam",
                    preProc = c("center", "scale"), 
                    trControl = controlobject)
summary(Default.PAM)
Default.PAM

confusionMatrix(Default.PAM)

plot(Default.PAM)

## Added tuning parameters
controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
nscGrid <- data.frame(.threshold = 0:25)
set.seed(590)
Default.PAM <- train(default ~ student + balance + income, data=trainingset,
                    method="pam",
                    preProc = c("center", "scale"),
                    tuneGrid=nscGrid,
                    trControl = controlobject)
summary(Default.PAM)
Default.PAM

confusionMatrix(Default.PAM)

plot(Default.PAM)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.QDA <- train(default ~ student + balance + income, data=trainingset,
                    method="qda",
                    preProc = c("center", "scale"),
                    trace=FALSE,
                    trControl = controlobject)
summary(Default.QDA)
Default.QDA

confusionMatrix(Default.QDA)

install.packages("klaR")

library(klaR)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.RDA <- train(default ~ student + balance + income, data=trainingset,
                    method="rda",
                    preProc = c("center", "scale"),
                    trace=FALSE,
                    trControl = controlobject)
summary(Default.RDA)
Default.RDA

confusionMatrix(Default.RDA)

plot(Default.RDA)

install.packages("mda")

library(mda)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.MDA <- train(default ~ student + balance + income, data=trainingset,
                    method="mda",
                    preProc = c("center", "scale"),
                    trControl = controlobject)
summary(Default.MDA)
Default.MDA

confusionMatrix(Default.MDA)

plot(Default.MDA)

## Added tuning parameters
controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.MDA <- train(default ~ student + balance + income, data=trainingset,
                    method="mda",
                    preProc = c("center", "scale"),
                    tuneGrid= expand.grid(.subclasses = 1:8),
                    trControl = controlobject)
summary(Default.MDA)
Default.MDA

confusionMatrix(Default.MDA)
plot(Default.MDA)

library (earth)
library (plotmo)
library (plotmix)
library (TeachingDemos)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.FDA <- train(default ~ student + balance + income, data=trainingset,
                    method="fda",
                    preProc = c("center", "scale"),
                    trace=FALSE,
                    trControl = controlobject)
summary(Default.FDA)
Default.FDA

confusionMatrix(Default.FDA)

plot(Default.FDA)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.NN <- train(default ~ student + balance + income, data=trainingset,
                    method="nnet",
                    preProc = c("center", "scale"),
                    trace=FALSE,
                    trControl = controlobject)
summary(Default.NN)
Default.NN

confusionMatrix(Default.NN)

plot(Default.NN)

install.packages("kernlab")

library(kernlab)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.SVM <- train(default ~ student + balance + income, data=trainingset, method="svmRadial",
                    trControl = controlobject)
summary(Default.SVM)

Default.SVM

confusionMatrix(Default.SVM)

plot(Default.SVM)

library(plotmo)

plotmo(Default.SVM)
plotres(Default.SVM)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.KNN <- train(default ~ student + balance + income, data=trainingset, method="knn",
                    trControl = controlobject)
summary(Default.KNN)
Default.KNN

confusionMatrix(Default.KNN)

plot(Default.KNN)

plotmo(Default.KNN)
plotres(Default.KNN)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.NB <- train(default ~ student + balance + income, data=trainingset,
                    method="nb",
                    preProc = c("center", "scale"),
                    trace=FALSE,
                    trControl = controlobject)
summary(Default.NB)
Default.NB

confusionMatrix(Default.NB)

plot(Default.NB)

library(rpart)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.CART <- train(default ~ student + balance + income, data=trainingset,
                    method="rpart",
                    preProc = c("center", "scale"),
                    trControl = controlobject)
summary(Default.CART)
Default.CART

confusionMatrix(Default.CART)

plot(Default.CART)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.CTree <- train(default ~ student + balance + income, data=trainingset,
                    method="ctree",
                    tuneLength = 10,
                    trControl = controlobject)
summary(Default.CTree)
show(Default.CTree)
Default.CTree

confusionMatrix(Default.CTree)

plot(Default.CTree)

library(ipred)
library(plyr)
library(e1071)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.Bag <- train(default ~ student + balance + income, data=trainingset,
                    method="treebag",
                    preProc = c("center", "scale"),
                    trControl = controlobject)
summary(Default.Bag)
Default.Bag

confusionMatrix(Default.Bag)

install.packages("randomForest")

library(randomForest)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.RF <- train(default ~ student + balance + income, data=trainingset,
                    method="rf",
                    preProc = c("center", "scale"),
                    trControl = controlobject)
summary(Default.RF)
Default.RF

confusionMatrix(Default.RF)

plot(Default.RF)

varImp(Default.RF)
plot(varImp(Default.RF))

plotmo(Default.RF)
plotres(Default.RF)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.Boost <- train(default ~ student + balance + income, data=trainingset,
                    method="gbm",
                    preProc = c("center", "scale"),
                    verbose=FALSE,
                    trControl = controlobject)
summary(Default.Boost)
Default.Boost

confusionMatrix(Default.Boost)

plot(Default.Boost)

## Added tuning parameters
## Takes very long to run
controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2),
                    .n.trees = seq(100, 1000, by =50),
                    .shrinkage  = c(.001, .1),
                        .n.minobsinnode = 10 )
set.seed(590)
Default.Boost <- train(default ~ student + balance + income, data=trainingset,
                    method="gbm",
                    preProc = c("center", "scale"),
                    tuneGrid = gbmGrid,
                    trControl = controlobject)
summary(Default.Boost)
Default.Boost

confusionMatrix(Default.Boost)

plot(Default.Boost)

install.packages("C50")

library(C50)

controlobject <- trainControl(method="repeatedcv", number = 10, repeats = 5)
set.seed(590)
Default.C5.0 <- train(default ~ student + balance + income, data=trainingset,
                    method="C5.0",
                    preProc = c("center", "scale"),
                    trace=FALSE,
                    trControl = controlobject)
summary(Default.C5.0)
Default.C5.0

confusionMatrix(Default.C5.0)

plot(Default.C5.0)

installed.packages()

install.packages("caret")

update.packages("caret")

library(caret)

ressum <- resamples(list("LDA" = Default.LDA, 
                         "Log" = Default.Log,
                         "PLS" = Default.PLS,
                         "PLR" = Default.PLR,
                         "PAM" = Default.PAM,
                         "QDA" = Default.QDA,
                         "RDA" = Default.RDA,
                         "MDA" = Default.MDA,
                         "FDA" = Default.FDA,
                         "NN" = Default.NN,
                         "SVM" = Default.SVM,
                         "KNN" = Default.KNN,
                         "NB" = Default.NB,
                         "CART" = Default.CART,
                         "CTree" = Default.CTree,
                         "Bag" = Default.Bag,
                         "RF" = Default.RF,
                         "Boost" = Default.Boost,
                         "C5.0" = Default.C5.0))

names(ressum)

ressum$metrics
ressum$values

summary(ressum)

parallelplot(ressum, metric="Accuracy")
parallelplot(ressum, metric="Kappa")

bwplot(ressum, metric="Accuracy")
bwplot(ressum, metric="Kappa")

densityplot(ressum, metric="Accuracy")
densityplot(ressum, metric="Kappa")

dotplot(ressum, metric="Accuracy")
dotplot(ressum, metric="Kappa")

splom(ressum, metric="Accuracy")
splom(ressum, metric="Kappa")

ressum1 <- summary(ressum)
names(ressum1)

names(ressum1$statistics)

ressum.Accuracy <- ressum1$statistics$Accuracy
ressum.Accuracy

ressum.Kappa <- ressum1$statistics$Kappa
ressum.Kappa

ressum.Accuracy <- ressum.Accuracy[ ,4]
ressum.Accuracy
ressum.Kappa <- ressum.Kappa[ ,4]
ressum.Kappa

ressum.train <- cbind(ressum.Accuracy, ressum.Kappa)
ressum.train

Default.LDA.pred <- predict(Default.LDA, testingset)
Default.LDA.test <- data.frame(obs = testingset$default, pred = Default.LDA.pred)
Default.LDA.stats <- defaultSummary(Default.LDA.test)
Default.LDA.stats

Default.Log.pred <- predict(Default.Log, testingset)
Default.Log.test <- data.frame(obs = testingset$default, pred = Default.Log.pred)
Default.Log.stats <- defaultSummary(Default.Log.test)
Default.Log.stats

Default.PLS.pred <- predict(Default.PLS, testingset)
Default.PLS.test <- data.frame(obs = testingset$default, pred = Default.PLS.pred)
Default.PLS.stats <- defaultSummary(Default.PLS.test)
Default.PLS.stats

Default.PLR.pred <- predict(Default.PLR, testingset)
Default.PLR.test <- data.frame(obs = testingset$default, pred = Default.PLR.pred)
Default.PLR.stats <- defaultSummary(Default.PLR.test)
Default.PLR.stats

Default.PAM.pred <- predict(Default.PAM, testingset)
Default.PAM.test <- data.frame(obs = testingset$default, pred = Default.PAM.pred)
Default.PAM.stats <- defaultSummary(Default.PAM.test)
Default.PAM.stats

Default.QDA.pred <- predict(Default.QDA, testingset)
Default.QDA.test <- data.frame(obs = testingset$default, pred = Default.QDA.pred)
Default.QDA.stats <- defaultSummary(Default.QDA.test)
Default.QDA.stats

Default.RDA.pred <- predict(Default.RDA, testingset)
Default.RDA.test <- data.frame(obs = testingset$default, pred = Default.RDA.pred)
Default.RDA.stats <- defaultSummary(Default.RDA.test)
Default.RDA.stats

Default.MDA.pred <- predict(Default.MDA, testingset)
Default.MDA.test <- data.frame(obs = testingset$default, pred = Default.MDA.pred)
Default.MDA.stats <- defaultSummary(Default.MDA.test)
Default.MDA.stats

Default.FDA.pred <- predict(Default.FDA, testingset)
Default.FDA.test <- data.frame(obs = testingset$default, pred = Default.FDA.pred)
Default.FDA.stats <- defaultSummary(Default.FDA.test)
Default.FDA.stats

Default.NN.pred <- predict(Default.NN, testingset)
Default.NN.test <- data.frame(obs = testingset$default, pred = Default.NN.pred)
Default.NN.stats <- defaultSummary(Default.NN.test)
Default.NN.stats

Default.SVM.pred <- predict(Default.SVM, testingset)
Default.SVM.test <- data.frame(obs = testingset$default, pred = Default.SVM.pred)
Default.SVM.stats <- defaultSummary(Default.SVM.test)
Default.SVM.stats

Default.KNN.pred <- predict(Default.KNN, testingset)
Default.KNN.test <- data.frame(obs = testingset$default, pred = Default.KNN.pred)
Default.KNN.stats <- defaultSummary(Default.KNN.test)
Default.KNN.stats

Default.NB.pred <- predict(Default.NB, testingset)
Default.NB.test <- data.frame(obs = testingset$default, pred = Default.NB.pred)
Default.NB.stats <- defaultSummary(Default.NB.test)
Default.NB.stats

Default.CART.pred <- predict(Default.CART, testingset)
Default.CART.test <- data.frame(obs = testingset$default, pred = Default.CART.pred)
Default.CART.stats <- defaultSummary(Default.CART.test)
Default.CART.stats

Default.CTree.pred <- predict(Default.CTree, testingset)
Default.CTree.test <- data.frame(obs = testingset$default, pred = Default.CTree.pred)
Default.CTree.stats <- defaultSummary(Default.CTree.test)
Default.CTree.stats

Default.Bag.pred <- predict(Default.Bag, testingset)
Default.Bag.test <- data.frame(obs = testingset$default, pred = Default.Bag.pred)
Default.Bag.stats <- defaultSummary(Default.Bag.test)
Default.Bag.stats

Default.RF.pred <- predict(Default.RF, testingset)
Default.RF.test <- data.frame(obs = testingset$default, pred = Default.RF.pred)
Default.RF.stats <- defaultSummary(Default.RF.test)
Default.RF.stats

Default.Boost.pred <- predict(Default.Boost, testingset)
Default.Boost.test <- data.frame(obs = testingset$default, pred = Default.Boost.pred)
Default.Boost.stats <- defaultSummary(Default.Boost.test)
Default.Boost.stats

Default.C5.0.pred <- predict(Default.C5.0, testingset)
Default.C5.0.test <- data.frame(obs = testingset$default, pred = Default.C5.0.pred)
Default.C5.0.stats <- defaultSummary(Default.C5.0.test)
Default.C5.0.stats

ressum.test <- data.frame(Default.LDA.stats,
                         Default.Log.stats,
                         Default.PLS.stats,
                         Default.PLR.stats, 
                         Default.PAM.stats,
                         Default.QDA.stats,
                         Default.RDA.stats,
                         Default.MDA.stats,
                         Default.FDA.stats,
                         Default.NN.stats,
                         Default.SVM.stats,
                         Default.KNN.stats,
                         Default.NB.stats,
                         Default.CART.stats,
                         Default.CTree.stats,
                         Default.Bag.stats,
                         Default.RF.stats,
                         Default.Boost.stats,
                         Default.C5.0.stats)
ressum.test

ressum.test1 <- t(ressum.test)
ressum.test1

str(ressum.test1)

ressum.test1 <- data.frame(ressum.test1)
str(ressum.test1)

ressum.test1$Model <- row.names(ressum.test1)
row.names(ressum.test1) <- seq(1:19)
ressum.test1$Model <- gsub(".stats", "", ressum.test1$Model)
ressum.test1$Model <- gsub("Default.", "", ressum.test1$Model)
ressum.test1$Model <- format(ressum.test1$Model, justify="left")
ressum.test1[ , c(3, 1:2)]

ressum.test.Accuracy <- ressum.test1[order(ressum.test1$Accuracy, decreasing = TRUE), ]
ressum.test.Accuracy[ , c(3, 1:2)]

ressum.test.Kappa <- ressum.test1[order(ressum.test1$Kappa, decreasing = TRUE), ]
ressum.test.Kappa[ , c(3, 1:2)]

ggplot(ressum.test1,aes(x= reorder(Model,Accuracy), y=Accuracy))+
geom_bar(stat ="identity") + coord_flip()

ggplot(ressum.test1,aes(x= reorder(Model,Kappa), y=Kappa))+
geom_bar(stat ="identity") + coord_flip()
