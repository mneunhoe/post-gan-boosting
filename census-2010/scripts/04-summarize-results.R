source("scripts/00-setup.R")
source("../zz-functions.R")
source("../post_GAN_functions.R")

# Load synthetic samples and real data

# Get training data
gan_list <- readRDS("gan-input/gan_list.RDS")

data <- as.matrix(gan_list$input_z)

gan_list$input <- NULL
gan_list$input_z <- NULL

orig_data <- synth_to_orig(data, gan_list)

tmp_data <- readRDS("synthetic-output/res_df.RDS")

# Calculate pMSE

pmse_dp <-
  lapply(tmp_data, function(x)
    gan_pmse(
      x,
      orig_data,
      method = "cart",
      cp = 0.00001,
      maxorder = 0,
      n = 4000
    ))


pmse_dp$sample$utilR
round(sapply(pmse_dp, function(x) x$utilR), 3)

my_table <-
  table(orig_data$race, orig_data$hispanic, orig_data$female)

order_hist <- order(-as.numeric(my_table) / sum(my_table))

hist_num <- (as.numeric(my_table) / sum(my_table))[order_hist]

hist_num_synth <- lapply(tmp_data, function(tmp) {
  tmp_table <- table(tmp$race, tmp$hispanic, tmp$female)
  
  hist_num_synth <-
    (as.numeric(tmp_table) / sum(tmp_table))[order_hist]
  return(hist_num_synth)
})


# Calculate the accuracies and the orders for each cell of the three-way marginal table
accuracies <- matrix(NA, 4, 550)
for(i in 1:4){
  accuracies[i,] <-  1 - abs(hist_num_synth[[i]] - hist_num)
}



average_accuracies <- round(rowSums(accuracies)/550,4)


ord_acc <- apply(-accuracies, 2, order)

best_accuracy <- c(sum(ord_acc[1,] == 1),
  sum(ord_acc[2,] == 1),
  sum(ord_acc[3,] == 1),
  sum(ord_acc[4,] == 1))
