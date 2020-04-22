source("scripts/00-setup.R")
source("../zz-functions.R")

# Load synthetic samples
tmp <- readRDS("synthetic-output/res_df.RDS")

tmp_data_dp <- tmp$dp
tmp_data_nodp <- tmp$nodp

# Load real data
gan_list <- readRDS("gan-input/gan_list.RDS")

data <- as.matrix(gan_list$input_z)

gan_list$input <- NULL
gan_list$input_z <- NULL

orig_data <- synth_to_orig(data, gan_list)

# Create matrix for the grouped bar plot
create_bp_mat <- function(orig_data, samples, variable = "race") {
  bp_mat <-
    cbind(table(orig_data[, variable]) / sum(table(orig_data[, variable])),
          sapply(samples, function(run)
            lapply(run, function(m)
              sapply(m, function(x)
                table(x[, variable]) / sum(table(x[, variable])))))[[1]])
  colnames(bp_mat) <- c("Real Data", "GAN", "DRS", "PGB", "PGB+DRS")
  
  return(bp_mat)
}

png(
  "figures/Figure3.png",
  width = 784,
  height = 657,
  units = "px"
)
par(mar = c(8, 4, 3, 0))
x <-
  barplot(
    t(create_bp_mat(orig_data, tmp_data_dp, "race")),
    beside = T,
    ylab = "Proportion in Sample",
    ylim = c(0, 1),
    las = 2,
    col = c("grey", viridis(4)),
    border = "white",
    main = "Distribution of Race Membership in Synthetic 1940 Census Samples",
    xaxt = "n"
  )

labs <-
  c(
    "White",
    "Black/African American",
    "American Indian or Alaska Native",
    "Chinese",
    "Japanese",
    "Other Asian or Pacific Islander"
  )
text(
  cex = 1,
  x = x[5,] - .25,
  y = -0.03,
  labs,
  xpd = TRUE,
  srt = 33,
  adj = 1
)
legend(
  "topright",
  c("Real Data", "DP GAN", "DP DRS", "DP PGB", "DP PGB+DRS"),
  fill = c("grey", viridis(4)),
  bty = "n",
  border = "white"
)
dev.off()


# Calculate pMSE scores

cart_pmse_dp <-
  lapply(tmp_data_dp, function(run)
    lapply(run, function(x)
      lapply(x, function(y)
        gan_pmse(y, orig_data, cp = 0.001))))


lapply(cart_pmse_dp, function(run)
  lapply(run, function(m)
    sapply(m, function(x)
      x$utilStd)))
lapply(cart_pmse_dp, function(run)
  lapply(run, function(m)
    sapply(m, function(x)
      x$utilR)))

cart_pmse_nodp <-
  lapply(tmp_data_nodp, function(run)
    lapply(run, function(x)
      lapply(x, function(y)
        gan_pmse(y, orig_data, cp = 0.001))))

lapply(cart_pmse_nodp, function(run)
  lapply(run, function(m)
    sapply(m, function(x)
      x$utilStd)))

lapply(cart_pmse_nodp, function(run)
  lapply(run, function(m)
    sapply(m, function(x)
      x$utilR)))


logit_pmse_dp <-
  lapply(tmp_data_dp, function(run)
    lapply(run, function(x)
      lapply(x, function(y)
        gan_pmse(y, orig_data, method = "logit", maxorder = 0))))


lapply(logit_pmse_dp, function(run)
  lapply(run, function(m)
    sapply(m, function(x)
      x$utilStd)))
lapply(logit_pmse_dp, function(run)
  lapply(run, function(m)
    sapply(m, function(x)
      x$utilR)))


logit_pmse_nodp <-
  lapply(tmp_data_nodp, function(run)
    lapply(run, function(x)
      lapply(x, function(y)
        gan_pmse(y, orig_data, method = "logit", maxorder = 0))))


lapply(logit_pmse_nodp, function(run)
  lapply(run, function(m)
    sapply(m, function(x)
      x$utilStd)))
lapply(logit_pmse_nodp, function(run)
  lapply(run, function(m)
    sapply(m, function(x)
      x$utilR)))


png(
  "figures/Figure2.png",
  width = 784,
  height = 657,
  units = "px"
)
plot(
  density(orig_data$age),
  col = "grey",
  lwd = 2,
  bty = "n",
  yaxt = "n",
  ylab = "",
  xlab = "Age"
  ,
  ylim = c(0, 0.05),
  xlim = c(-20, 150),
  main = "Marginal Distributions of Age in Synthetic 1940 Census Samples"
)
for (i in 1:4) {
  lines(density(tmp_data_dp[[1]][[1]][[i]]$age),
        col = viridis(4)[i],
        lwd = 2)
}
legend(
  "topright",
  c("Real Data", "DP GAN", "DP DRS", "DP PGB", "DP PGB+DRS"),
  col = c("grey", viridis(4)),
  bty = "n",
  lty = "solid",
  lwd = 2
)
dev.off()

# Same plot without privacy

# plot(density(orig_data$age), col = "grey", lwd = 2, bty = "n", yaxt = "n", ylab = "", xlab = "Age"
#      , ylim = c(0, 0.05), xlim = c(-20, 150), main = "Marginal Distributions of Age in Synthetic 1940 Census Samples"
# )
# for(i in 1:4){
#   lines(density(tmp_data_nodp[[1]][[1]][[i]]$age), col = viridis(4)[i], lwd = 2)
# }
# legend("topright", c("Real Data", "GAN", "DRS", "PBG", "PBG+DRS"), col = c("grey", viridis(4)), bty = "n", lty = "solid", lwd =2)


# Load original training and test data
train_test <- readRDS("pums1940/processed-data/train_test.RDS")

train_df <- synth_to_orig(train_test$train$input_z, gan_list)
test_df <- synth_to_orig(train_test$test$input_z, gan_list)

# Define all three-way regression combinations
reg_combs <- combn(colnames(orig_data[, c(-4, -6, -8)]), 3)

# Function to calculate regressions and collect measures
get_regressions <- function(data, orig_data) {
  coef_real <- list()
  coef_synth <- list()
  pred_real <- list()
  se_real <- list()
  
  pred_synth <- list()
  reg_real <- list()
  reg_synth <- list()
  se_synth <- list()
  
  for (i in 1:ncol(reg_combs)) {
    ff <-
      as.formula(paste("inc", paste(reg_combs[1:3, i], collapse = " + "), sep =
                         " ~ "))
    
    tmp_real <- lm(ff, data.frame(orig_data))
    
    reg_real[[i]] <- tmp_real
    coef_real[[i]] <- coef(tmp_real)
    se_real[[i]] <- summary(tmp_real)$coefficients[, "Std. Error"]
    pred_real[[i]] <-
      predict(tmp_real, newdata = data.frame(test_df))
    
    
    tmp_synth <- lm(ff, data.frame(data))
    
    reg_synth[[i]] <- tmp_synth
    coef_synth[[i]] <- coef(tmp_synth)
    se_synth[[i]] <- summary(tmp_synth)$coefficients[, "Std. Error"]
    pred_synth[[i]] <-
      predict(tmp_synth, newdata = data.frame(test_df))
    cat(i, "\n")
  }
  
  res <- list(pred_synth, pred_real)
  return(res)
  
}

# Get out of sample errors
reg_dp <-
  lapply(tmp_data_dp, function(run)
    lapply(run, function(m)
      lapply(m, function(x)
        get_regressions(x, orig_data))))


synth_errors_dp <- lapply(reg_dp, function(run)
  lapply(run, function(m)
    lapply(m, function(x)
      lapply(1:ncol(reg_combs), function(y)
        rmse(x[[1]][[y]], test_df$inc)))))


real_errors_dp <- lapply(reg_dp, function(run)
  lapply(run, function(m)
    lapply(m, function(x)
      lapply(1:ncol(reg_combs), function(y)
        rmse(x[[2]][[y]], test_df$inc)))))




error_mat <- cbind(unlist(synth_errors), unlist(real_errors))



a <- unlist(real_errors_dp)
b <- unlist(synth_errors_dp)

png(
  "figures/Figure4.png",
  width = 784,
  height = 657,
  units = "px"
)
par(mar = c(3, 4, 3, 3))
plot(
  1,
  1,
  ylim = c(min(c(a, b)), max(c(a, b))),
  xlim = c(0.5, 1.5),
  type = "n",
  bty = "n",
  xaxt = "n",
  xlab = "",
  las = 1,
  ylab = "RMSE in Income",
  main = "Regression RMSE with Synthetic 1940 Samples"
)

col_vec <- viridis::viridis(4)
lty_vec <- c("solid", "solid", "solid", "solid")
for (run in 1:runs) {
  for (m_run in 1:m) {
    for (model in 1:4) {
      a <- unlist(real_errors_dp[[run]][[m_run]][[model]])
      b <- unlist(synth_errors_dp[[run]][[m_run]][[model]])
      
      segments(
        x0 = rep(0.5, length(a)),
        y0 = a,
        x1 = rep(1.5, length(a)),
        y1 = b,
        col = adjustcolor(col_vec[model], alpha = 0.5),
        lty = lty_vec[model],
        lwd = 2
      )
      
    }
  }
}

legend(
  "topleft",
  legend = c("DP GAN", "DP DRS", "DP PGB", "DP PGB+DRS"),
  col = col_vec,
  lty = "solid",
  bty = "n",
  lwd = 2
)
axis(
  1,
  at = c(0.5, 1.5),
  labels = c("Trained on real data", "Trained on synthetic data")
)
dev.off()