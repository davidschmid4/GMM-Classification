

### Package mclust    https://cran.r-project.org/web/packages/mclust/vignettes/mclust.html
pacman::p_load(mclust, readr)

## DATA
##### DIGIT #################################################
semeion <- read.table("semeion.data", quote = "\"", comment.char = "") # put in right path here!
digit_data <- semeion[ , 1:256]
which_digit <- apply(semeion[ , 257:266], 1, function(x) which.max(x) - 1)
digit_data <- cbind(digit_data, which_digit)
# chose training and test set by chance         
set.seed(123)                                                             # for reproducibility
random <- sample(1:nrow(digit_data), 0.8 * nrow(digit_data)) # 80%: training data, 20%: test data
digit.train <- digit_data[random, ]
digit.test <- digit_data[-random, ]
rm(random, semeion, which_digit)
##### IONOSPHERE #############################################
iodata <- read_csv("ionosphere.data",col_names = FALSE)
# chose training and test set by chance                  
set.seed(123) # for reproducibility
random <- sample(1:nrow(iodata), 0.8 * nrow(iodata)) # 80%: training data, 20%: test data
iodata.train <- iodata[random, ]
iodata.test <- iodata[-random, ]
rm(random)
###### Wall Following Robot Dataset ##########################
robot2 <- read.table("sensor_readings_2.data", quote = "\"", comment.char = "", sep = ",") # put in right path here!
names(robot2)[3] <- "class"
robot4 <- read.table("sensor_readings_4.data", quote = "\"", comment.char = "", sep = ",") # put in right path here!
names(robot4)[5] <- "class"
robot24 <- read.table("sensor_readings_24.data", quote = "\"", comment.char = "", sep = ",") # put in right path here!
names(robot24)[25] <- "class"
###############################################################


## PREPARE DATA FOR GMM DA FUNCTION WITH CROSSVALIDATION
# io
iodat <- iodata[,-35]
ioclass <- iodata$X35
# digit
digitdat <- digit_data[, -257]
digitclass <- digit_data$which_digit
# robot
rob2dat <- robot2[,-3]
rob2class <- robot2$class
rob4dat <- robot4[, -5]
rob4class <- robot4$class
rob24dat <- robot24[, -25]
rob24class <- robot24$class


#-------------------------------------------------------------------------------------------------------

## PREPARE DATA FOR GMM DA FUNCTION WITHOUT CROSSVALIDATION   
iodata.train <- iodata.train[,-35]
iodata.test.class <- iodata.test$X35
iodata.test <- iodata.test[,-35]
digit.train.class <- digit.train$which_digit
digit.train <- digit.train[,-257]
digit.test.class <- digit.test$which_digit
digit.test <- digit.test[,-257]
###############################################################

### MclustDA EDDA Classification
########## DIGIT TRAINDATEN:
start_time <- Sys.time()
moddigit <- MclustDA(digit.train, digit.train.class, modelType = "EDDA")
end_time <- Sys.time()
end_time - start_time
#summary(moddigit)
##############  DIGIT TESTDATEN:
summary(moddigit, newdata = digit.test, newclass = digit.test.class)
## 10 fold
cvdig <- cvMclustDA(moddigit) # default 10-fold CV
cvdig[c("error", "se")]
summary(cvdig)

moddigit$prop
#-------------------------------------------------------------------------------------------------------
########## IONESPHERE TRAINDATEN:
start_time <- Sys.time()
modio <- MclustDA(iodata.train, iodata.train.class, modelType = "EDDA")
end_time <- Sys.time()
end_time - start_time
##############  IONOSPHERE TESTDATEN:
summary(modio, newdata = iodata.test, newclass = iodata.test.class)
## 10 fold
cviod <- cvMclustDA(modio) # default 10-fold CV
cviod[c("error", "se")]
library(caret)
train_control <- caret::trainControl(method = "cv", number = 10, summaryFunction = "mseSummary")


plot(modio, what = "density")
###############################################################

# EDDA imposes a single mixture component for each group. However, in certain circumstances
# more complexity may improve performance. A more general approach, called MclustDA, has been
# proposed by Fraley and Raftery (2002), where a finite mixture of Gaussian distributions is used within
# each class, with number of components and covariance matrix structures (expressed following the
# usual decomposition) being different between classes.









#-------------------------------------------------------------------------------------------------------


### MclustDA EDDA Classification
# stratified train-test split with caret::createDataPartition
data <- robot2
trainIndex <- caret::createDataPartition(data$class, p=.8, list=FALSE)
robot2train <- data[trainIndex,]
Class.train.robot2 <- robot2train$class
robot2test <- data[-trainIndex,]
Class.test.robot2 <- robot2test$class

robot2train <- robot2train[,-3]
robot2test <- robot2test[,-3]


########## train
start_time <- Sys.time()
modRobot2 <- MclustDA(robot2train, Class.train.robot2, modelType = "EDDA")
end_time <- Sys.time()
end_time - start_time

###############

# Test
summary(modRobot2, newdata = robot2test, newclass = Class.test.robot2)
## 10 fold
cviod <- cvMclustDA(modRobot2) # default 10-fold CV
cviod[c("error", "se")]
# create plot for MclustDA Robot Walk
drmod2 <- MclustDR(modRobot2)
summary(drmod2)
p <- plot(drmod2, what = "boundaries", ngrid = 200)
p


plot(modRobot2, what = "density", type = "hdr")
#-------------------------------------------------------------------------------------------------------


### MclustDA EDDA Classification
# stratified train-test split with caret::createDataPartition
data <- robot4
trainIndex <- caret::createDataPartition(data$class, p=.8, list=FALSE)
robot4train <- data[trainIndex,]
Class.train.robot4 <- robot4train$class
robot4test <- data[-trainIndex,]
Class.test.robot4 <- robot4test$class

robot4train <- robot4train[,-5]
robot4test <- robot4test[,-5]


########## train
start_time <- Sys.time()
modRobot4 <- MclustDA(robot4train, Class.train.robot4, modelType = "EDDA")
end_time <- Sys.time()
end_time - start_time

###############

# Test
summary(modRobot4, newdata = robot4test, newclass = Class.test.robot4)
## 10 fold
cviod <- cvMclustDA(modRobot4) # default 10-fold CV
cviod[c("error", "se")]
# create plot for MclustDA Robot Walk
drmod2 <- MclustDR(modRobot4)
summary(drmod2)
p <- plot(drmod2, what = "boundaries", ngrid = 200)
p


#-------------------------------------------------------------------------------------------------------

#####
# 24 Sensor input dataset
####
robot24 <- read.table("WallRobot/sensor_readings_24.data", quote = "\"", comment.char = "", sep = ",") # put in right path here!
names(robot24)[25] <- "class"
str(robot24)

### MclustDA EDDA Classification
# stratified train-test split with caret::createDataPartition
data <- robot24
trainIndex <- caret::createDataPartition(data$class, p=.8, list=FALSE)
robot24train <- data[trainIndex,]
Class.train.robot24 <- robot24train$class
robot24test <- data[-trainIndex,]
Class.test.robot24 <- robot24test$class

robot24train <- robot24train[,-25]
robot24test <- robot24test[,-25]


########## train
start_time <- Sys.time()
modRobot24 <- MclustDA(robot24train, Class.train.robot24, modelType = "EDDA")
end_time <- Sys.time()
end_time - start_time


###############

# Test
summary(modRobot24, newdata = robot24test, newclass = Class.test.robot24)
## 10 fold
cviod <- cvMclustDA(modRobot24) # default 10-fold CV
cviod[c("error", "se")]
# create plot for MclustDA Robot Walk
drmod2 <- MclustDR(modRobot24)
summary(drmod2)
p <- plot(drmod2, what = "boundaries", ngrid = 200)
p


#-------------------------------------------------------------------------------------------------------












































### MclustDA more complex:
########## DIGIT TRAINDATEN:
moddigit2 <- MclustDA(digit.train, digit.train.class)    #### more complex is not working for digit data
##############  DIGIT TESTDATEN:
## summary(moddigit2, newdata = digit.test, newclass = digit.test.class)
########## IONESPHERE TRAINDATEN:
modio2 <- MclustDA(iodata.train, iodata.train.class)    ### more complex is working but not bether than the first
##############  DIGIT TESTDATEN:
summary(modio2, newdata = iodata.test, newclass = iodata.test.class)
###############################################################


### --> in unserem Fall ist es also so, dass die Komponentenanzahl und die Kovarianzstruktur 
### über die Klassen hinweg gleich ist. Dies ergibt das bessere Modell!

 plot(modio, what = "classification", newdata = iodata.test )
 plot(modio, what = "scatterplot", dimens = c(2,3))
 plot(modio, what = "scatterplot", dimens = c(3,1))

 
 # Another interesting graph can be obtained by projecting the data on a dimension reduced subspace
 # (Scrucca, 2014) with the commands:
   
 drmod2 <- MclustDR(modio)
 plot(drmod2, what = "boundaries", ngrid = 200)
 