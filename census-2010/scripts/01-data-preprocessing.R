source("scripts/00-setup.R")
source("../zz-functions.R")

# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).

if (!require("ipumsr"))
  stop(
    "Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')"
  )

ddi <- read_ipums_ddi("raw-data/usa_00007.xml")
data <- read_ipums_micro(ddi)

# Subset to California
data_cal <- data[data$US2010G_STATE == "06",]
rm(data)
# Collect attributes for training data
data_gan <- data.frame(matrix(nrow = nrow(data_cal)))

data_gan$female <- as.numeric(data_cal$US2010G_SEX)-1
data_gan$age <- as.numeric(data_cal$US2010G_AGE)

data_gan$matrix.nrow...nrow.data_cal.. <- NULL

data_gan$hispanic <- factor(data_cal$US2010G_HISPAN)

data_gan$race <- factor(data_cal$US2010G_RACESHORT)

data_gan$puma <- factor(data_cal$US2010G_PUMA)

rm(data_cal)
# Define types for gumbel-softmax trick
types <- c("bin", "num", "fac", "fac", "fac")

# Create and store gan input files
gan_list <- gan_input(data_gan, type = types)
saveRDS(gan_list, "gan-input/gan_list.RDS")

data.table::fwrite(gan_list$input_z, file = "gan-input/gan_input.csv", verbose = T)
