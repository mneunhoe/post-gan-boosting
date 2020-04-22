source("scripts/00-setup.R")

# Define Variance-Covariance matrix for each cluster
sigma_mat <- matrix(c(0.05, 0, 0, 0.05)^2, nrow = 2, ncol = 2)

# Define Cluster Means
means_mat <- as.matrix(expand.grid(seq(-4, 4, by = 2), seq(-4, 4, by = 2)))

# Function to draw samples at one cluster mean
sample_2d_gaussian <- function(x, means_mat, sigma_mat, n = 1000) {
  mvrnorm(n, means_mat[x,], Sigma = sigma_mat)
}


set.seed(20200206)

# Create matrix with 1,000 observations at each of the 25 clusters
gaussian_df <- do.call("rbind", lapply(1:25, function(x) sample_2d_gaussian(x, means_mat, sigma_mat)))


# Store the raw data
write.csv(gaussian_df, "raw-data/gaussian_df.csv", row.names = F)