source("scripts/00-setup.R")
source("../zz-functions.R")

# Load full titanic data set from http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv
df <- read_csv("raw-data/titanic3.csv")

# Load kaggle titanic training set
df_train <- read_csv("raw-data/train.csv")

# Subset to relevant columns 
df <- df[,c(2,1,3,4,5,6,7,8,9,10,11)]

colnames(df) <- colnames(df_train)[2:12]

# Load kaggle titanic test set
df_test <- read_csv("raw-data/test.csv")

# Merge real outcome to kaggle test set
df_test <- merge(df_test, df, 
      all.x = T, all.y = F)

# Pre-process factor variables
df_train %<>% 
  mutate_at(vars(Survived, Pclass, Sex, Embarked, SibSp, Parch), factor)

df_test %<>% 
  mutate_at(vars(Survived, Pclass, Sex, Embarked, SibSp, Parch), factor)


# Define formula to get model frame for imputation
ff <- Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked

set.seed(200130)

# Impute missing values with missForest
# Independently done for train and test set

# Training set
rf_train <- model.frame(ff, df_train, na.action = na.pass)

rf_train_imputed <- missForest(rf_train)

# test set
rf_test <-model.frame(ff, df_test, na.action = na.pass)

rf_test_imputed <- missForest(rf_test)

# Make sure factors have same levels in train and test
levels(rf_train_imputed$ximp$Parch) <- levels(rf_test_imputed$ximp$Parch)

# Define types for gumbel-softmax trick
types <- c("fac", "fac", "fac", "num", "fac", "fac", "num", "fac")

# Create and store gan input files
gan_list <- gan_input(rf_train_imputed$ximp, type = types)

saveRDS(gan_list, "gan-input/gan_list.RDS")
data.table::fwrite(gan_list$input_z, file = "gan-input/gan_input.csv", verbose = T)

# Store processed test data
saveRDS(rf_test_imputed$ximp, "processed-data/test_data.RDS")
