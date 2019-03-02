library(caret)
library(randomForest)
library(tidyverse)
library(xgboost)

calc_accuracy=function(act,pred){
  rightIndexes=which(act==pred)
  accuracy=(length(rightIndexes)/length(act))*100
  return(accuracy)
}

setwd("C:/Users/Peters_K20/Documents/Titanic")
trainData=read.csv(file="C:/Users/Peters_K20/Documents/Titanic/train.csv", check.names=FALSE)
testData=read.csv(file="C:/Users/Peters_K20/Documents/Titanic/test.csv", check.names=FALSE)
testData$Survived=0
fullData=bind_rows(trainData,testData)


testData=trainData[751:length(rownames(trainData)),]
trainData=trainData[1:750,]

fullData=bind_rows(trainData,testData)




trainrows=1:length(row.names(trainData))
testrows=(length(row.names(trainData))+1):(length(row.names(fullData)))
target=as.factor(trainData$Survived)
theTarget="Survived"




#check out data
head(fullData)
colnames(fullData)
summary(fullData)
dim(fullData)

#check out classes of df vars
sapply(fullData,class)
fullData$Survived=as.factor(fullData$Survived)
fullData$Pclass=as.factor(fullData$Pclass)
fullData$Embarked=as.factor(fullData$Embarked)

#fix na values with median
sapply(fullData,function(col)sum(is.na(col)))
summary(trainData$Fare)
ggplot(trainData, aes(x=trainData$Fare, color=trainData$Survived))+
  geom_histogram(breaks=seq(1, 500, by = 10),binwidth=.5)+xlab("Fare")+ylab("Total Count")
naIndexes=which(is.na(fullData$Fare))
fullData[naIndexes,"Fare"]=median(fullData[-naIndexes,"Fare"])


#reinitiate train and test set
trainData=fullData[trainrows,]
testData=fullData[testrows,]

#name var
fullData$Name=as.character(fullData$Name)
title=sapply(strsplit(fullData$Name,"\\."), `[`, 1)
title=sapply(strsplit(title,"\\,"), `[`, 2)
title=trimws(title)
fullData$Title=title

ggplot(trainData, aes(x=fullData$Title[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("Title")+ylab("Total Count")

fullData[which(title=="Mrs"),"Title"]="Women"
fullData[which(title=="Master"),"Title"]="Boy"
fullData[which(title=="Miss"),"Title"]="Girl"
fullData[which(title=="Mlle"|title=="Mme"|title=="Ms"),"Title"]="Lady"
fullData[which(title=="Capt"|title=="Don"|title=="Col"|title=="Major"|title=="Dr"|title=="Rev"|
                 title=="Dona"|title=="Jonkheer"|title=="the Countess"|title=="Sir")
         ,"Title"]="Special"

ggplot(trainData, aes(x=fullData$Title[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("Title")+ylab("Total Count")

ggplot(trainData, aes(x=trainData$Age, color=trainData$Survived))+
  geom_histogram(breaks=seq(1, 100, by = 5),binwidth=.5)+xlab("Age")+ylab("Total Count")

titles=unique(fullData$Title)
for(i in 1:length(titles)){
  indexes=which(is.na(fullData$Age)&titles[i]==fullData$Title)
  indexes2=which(!is.na(fullData$Age)&titles[i]==fullData$Title)
  fullData[indexes,"Age"]=median(fullData[indexes2,"Age"])
}
  

#reinitiate train and test set
trainData=fullData[trainrows,]
testData=fullData[testrows,]



ggplot(trainData, aes(x=trainData$Pclass, color=trainData$Survived))+
  geom_bar()+xlab("Pclass")+ylab("Total Count")
ggplot(trainData, aes(x=trainData$Sex, color=trainData$Survived))+
  geom_bar()+xlab("Sex")+ylab("Total Count")
ggplot(trainData, aes(x=trainData$SibSp, color=trainData$Survived))+
  geom_bar()+xlab("SibSp")+ylab("Total Count")
ggplot(trainData, aes(x=trainData$Parch, color=trainData$Survived))+
  geom_bar()+xlab("Parch")+ylab("Total Count")
ggplot(trainData, aes(x=trainData$Embarked, color=trainData$Survived))+
  geom_bar()+xlab("Embarked")+ylab("Total Count")

summary(trainData$Age)
ggplot(trainData, aes(x=trainData$Age, color=trainData$Survived))+
  geom_histogram(breaks=seq(1, 80, by = 5),binwidth=.5)+xlab("Parch")+ylab("Total Count")
summary(trainData$Fare)
ggplot(trainData, aes(x=trainData$Fare, color=trainData$Survived))+
  geom_histogram(breaks=seq(0, 512, by = 20),binwidth=.5)+xlab("Fare")+ylab("Total Count")



#feature engineering

#bin fare and age
summary(fullData$Age)

fullData$AgeCat<-as.numeric(as.factor(cut(fullData$Age, c(0,11,18,22,26,33,38,60,81), right=FALSE)))

ggplot(trainData, aes(x=fullData$AgeCat[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("AgeCat")+ylab("Total Count")


summary(fullData$Fare)
fullData$FareCat<-as.numeric(as.factor(cut(fullData$Fare, c(0,9,26,35,70,250,550), right=FALSE)))

ggplot(trainData, aes(x=fullData$FareCat[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("FareCat")+ylab("Total Count")




  
  




#cabin var
fullData$Cabin=as.character(fullData$Cabin)
fullData[which(fullData$Cabin==""),"Cabin"]="Z0"
deck=substr(fullData$Cabin,0,1)
fullData$Deck=as.numeric(as.factor(deck))
fullData$Cabin_Num=sapply(strsplit(fullData$Cabin," "), length)

ggplot(trainData, aes(x=fullData$Deck[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("Deck")+ylab("Total Count")
ggplot(trainData, aes(x=fullData$Cabin_Num[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("Cabin_Num")+ylab("Total Count")





fullData$relatives=fullData$SibSp+fullData$Parch
relatives=ifelse(fullData$relatives==0,1,fullData$relatives)
fullData$farePerPerson=fullData$FareCat/relatives
fullData$ageClass=fullData$AgeCat*as.numeric(fullData$Pclass)

ggplot(trainData, aes(x=fullData$relatives[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("relatives")+ylab("Total Count")

ggplot(trainData, aes(x=fullData$ageClass[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("ageClass")+ylab("Total Count")


ggplot(trainData, aes(x=fullData$farePerPerson[trainrows], color=trainData$Survived))+
  geom_histogram(breaks=seq(0, 6, by = .5),binwidth=.5)+xlab("farePerPerson")+ylab("Total Count")


fullData$relative_cabin=fullData$relatives*fullData$Cabin_Num
ggplot(trainData, aes(x=fullData$relative_cabin[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("relative_cabin")+ylab("Total Count")


fullData$Alone=0
fullData[which(fullData$relatives==0),"Alone"]=1
ggplot(trainData, aes(x=fullData$Alone[trainrows], color=trainData$Survived))+
  geom_bar()+xlab("Alone")+ylab("Total Count")


#embarked
fullData$Embarked=as.numeric(fullData$Embarked)
fullData$Sex=as.numeric(fullData$Sex)
fullData$Pclass=as.numeric(fullData$Pclass)



fullData$Title=as.numeric(as.factor(fullData$Title))



#create vars based on ticket
fullData$GroupSize=0
fullData$AvgFare=0
fullData$FamilySurvival=0
for(i in 1:length(rownames(fullData))){
  ticket=fullData[i,"Ticket"]
  indexes=which(fullData$Ticket==ticket)
  fullData[i,"GroupSize"]=length(indexes)
  fullData[i,"AvgFare"]=fullData[i,"Fare"]/length(indexes)
  
  #can't include the index we are on in the survival var creation
  indexes=indexes[which(indexes!=i)]
  fullData[i,"FamilySurvival"]=mean(as.numeric(fullData[indexes[-i],"Survived"]))
}
ggplot(fullData, aes(x=fullData$FamilySurvival, color=fullData$Survived))+
  geom_histogram(breaks=seq(1, 2, by = .1),binwidth=.5)+xlab("FamilySurvival")+ylab("Total Count")
naIndex=which(is.na(fullData$FamilySurvival))
fullData[naIndex,"FamilySurvival"]=median(fullData[-naIndex,"FamilySurvival"])

#reinitiate train and test set
trainData=fullData[trainrows,]
testData=fullData[testrows,]







#rf model
set.seed(1235)
trainControl = trainControl(method = "cv", number = 10,verboseIter = TRUE)
#"Alone","AgeCat","FareCat","Embarked","Alone","relatives","Deck","Cabin_Num","SibSp","Parch"
#"GroupSize","AvgFare","FamilySurvival"
features=c("Pclass","Sex","farePerPerson","Title","GroupSize","AvgFare","FamilySurvival")
length(features)

rfModel = train(x = trainData[,features], y = target,method = "rf",trControl = trainControl,
                preProc = c("center", "scale"), metric = "Accuracy",tuneLength = 100)
rfModel
varImp(rfModel)
rfPrediction<- predict(rfModel,fullData[,features])

calc_accuracy(as.numeric(fullData[trainrows,theTarget]),as.numeric(rfPrediction[trainrows]))
calc_accuracy(as.numeric(fullData[testrows,theTarget]),as.numeric(rfPrediction[testrows]))


output=data.frame(PassengerId=testData$PassengerId,Survived=rfPrediction[testrows])
write.csv(output,file="C:/Users/Peters_K20/Documents/Titanic/Titanic_Stack3.csv",row.names=FALSE)


stack1=read.csv(file="C:/Users/Peters_K20/Documents/Titanic/Titanic_Stack1.csv", check.names=FALSE)
stack2=read.csv(file="C:/Users/Peters_K20/Documents/Titanic/Titanic_Stack2.csv", check.names=FALSE)
stack3=read.csv(file="C:/Users/Peters_K20/Documents/Titanic/Titanic_Stack3.csv", check.names=FALSE)
stack4=read.csv(file="C:/Users/Peters_K20/Documents/Titanic/Titanic_Stack4.csv", check.names=FALSE)
stack5=read.csv(file="C:/Users/Peters_K20/Documents/Titanic/Titanic_Stack5.csv", check.names=FALSE)

final=stack1$Survived+stack2$Survived+stack3$Survived+stack4$Survived+stack5$Survived
finalPreds=ifelse(final>2.5,1,0)
output=data.frame(PassengerId=testData$PassengerId,Survived=finalPreds)
write.csv(output,file="C:/Users/Peters_K20/Documents/Titanic/Stack_Output.csv",row.names=FALSE)
