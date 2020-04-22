gaussian_df <- read.csv("raw-data/gaussian_df.csv")

# Z-transform input
means <- apply(gaussian_df, 2, mean)

sds <- apply(gaussian_df, 2, sd)

gaussian_df_z <- sweep(sweep(gaussian_df, 2, means), 2, sds, "/")

# Store the GAN input data
write.csv(gaussian_df_z, "gan-input/gaussian_df_z.csv", row.names = F)