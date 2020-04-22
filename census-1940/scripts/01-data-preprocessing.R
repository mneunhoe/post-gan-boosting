source("scripts/00-setup.R")
source("../zz-functions.R")

# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).

if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")

ddi <- read_ipums_ddi("raw-data/usa_00006.xml")
data <- read_ipums_micro(ddi)

# Subset data to California
data_cal <- data[data$STATEFIP == 6, ]

# Subset to only adults
data_cal <- data_cal[data_cal$AGE>=18,]

# Make missing incomes missing
data_cal$INCWAGE[data_cal$INCWAGE >= 999998] <- NA

data <- as.matrix(data)

# Collect attributes in data frame to work with it
df <-
  data.frame(
    sex = data_cal$SEX - 1,
    age = data_cal$AGE,
    educ = as.factor(data_cal$EDUC),
    inc = data_cal$INCWAGE,
    race = as.factor(data_cal$RACE),
    hispan = as.factor(data_cal$HISPAN),
    marital = as.factor(data_cal$MARST),
    cty = as.factor(data_cal$COUNTYICP)
  )

# Remove Missings
df <- na.omit(df)

# Define types for gumbel softmax trick
df_type <- c("bin", "num", "fac", "num", "fac", "fac", "fac", "fac")

# Create Training and Test data sets
train_sel <- sample(nrow(df), floor(0.8*nrow(df)))

# Create gan_input object
gan_list <- gan_input(df[train_sel, ], df_type)
gan_test <- gan_input(df[-train_sel, ], df_type, means = gan_list$means, sds = gan_list$sds)

# Save the gan training data
saveRDS(gan_list, "gan-input/gan_list.RDS")

# Write csv file for gan training
data.table::fwrite(gan_list$input_z, file = "gan-input/gan_input.csv", verbose = T)

# Save training and test data
saveRDS(list(train = gan_list, test = gan_test), "processed-data/train_test.RDS")

