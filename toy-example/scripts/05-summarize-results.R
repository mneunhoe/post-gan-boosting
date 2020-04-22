source("scripts/00-setup.R")
source("../zz-functions.R")

data <-
  read.csv(
    "gan-input/gaussian_df_z.csv",
    header = T
  )
data <- as.matrix(data)


# Get raw data to transform samples back to original scale
data_raw <-
  as.matrix(read.csv("raw-data/gaussian_df.csv"))


means <- apply(data_raw, 2, mean)
sds <- apply(data_raw, 2, sd)

trans_back <- function(data, means, sds) {
  return(sweep(sweep(data, 2, sds, "*"), 2, means, "+"))
}

res_dp <- readRDS("synthetic-output/res_dp.RDS")

res_nodp <- readRDS("synthetic-output/res_nodp.RDS")

# Define functions to generate plots
setup_plot <- function(data, comparisons = 2, axis_limits = c(-6, 6)) {
  par(mfrow = c(comparisons, 5), oma = c(0, 0, 0, 0))
}

GAN_plots <-
  function(res,
           axis_limits = c(-6, 6),
           titles = c("last Generator", "DRS", "PGB", "PGB+DRS")) {
    
    plot(
      0,
      0,
      type = "n",
      pch = 19,
      col = adjustcolor("black", alpha = 0.1),
      xlim = axis_limits,
      ylim = axis_limits,
      bty = "n",
      las = 1,
      yaxt = "n",
      xaxt = "n",
      ylab = "",
      xlab = "",
      main = "Real Samples"
    )
    
    tmp_data <- trans_back(data, means, sds)
    points(tmp_data[, 1],
           tmp_data[, 2],
           pch = 19,
           col = adjustcolor("black", alpha = 0.1))
    abline(v = seq(-4, 4, 2), col = adjustcolor("grey", alpha = 0.3))
    abline(h = seq(-4, 4, 2), col = adjustcolor("grey", alpha = 0.3))
    
    for (i in 1:4) {
      plot(
        0,
        0,
        type = "n",
        pch = 19,
        col = adjustcolor("black", alpha = 0.1),
        xlim = axis_limits,
        ylim = axis_limits,
        bty = "n",
        las = 1,
        yaxt = "n",
        xaxt = "n",
        ylab = "",
        xlab = "",
        main = titles[i]
      )
      
      tmp_data <- trans_back(res[[i]], means, sds)
      
      if (nrow(tmp_data) < nrow(data)) {
        sample_sel_plot <- 1:nrow(tmp_data)
      } else {
        sample_sel_plot <- sample(nrow(tmp_data), nrow(data), replace = F)
      }
      
      
      points(tmp_data[sample_sel_plot, 1],
             tmp_data[sample_sel_plot, 2],
             pch = 19,
             col = adjustcolor("black", alpha = 0.1))
      abline(v = seq(-4, 4, 2), col = adjustcolor("grey", alpha = 0.3))
      abline(h = seq(-4, 4, 2), col = adjustcolor("grey", alpha = 0.3))
    }
  }


titles_nodp <-
  c("GAN Samples", "DRS Samples", "PGB Samples", "PGB+DRS Samples")

titles_dp <-
  c(expression(bold(paste(
    "DP GAN Samples ", epsilon, " = 0.635"
  ))),
  expression(bold(paste(
    "DP DRS Samples ", epsilon, " = 0.635"
  ))),
  expression(bold(paste(
    "DP PGB Samples ", epsilon, " = 1"
  ))),
  expression(bold(
    paste("DP PGB+DRS Samples ", epsilon, " = 1")
  )))

# Produce Figure like in the paper

png("figures/toy-example.png", width = 1000, height = 500, units = "px")
setup_plot(data, comparisons = 2)
GAN_plots(res_nodp, titles = paste(titles_nodp))
GAN_plots(res_dp, titles = titles_dp)
dev.off()

# Calculate quality scores and pMSE Scores

define_ellipse <- function(x_center, y_center, sigma_mat, p = 0.95){
  xc <- x_center # center x_c or h
  yc <- y_center # y_c or k
  a <- sigma_mat[1,1] * sqrt(qchisq(p, 2)) # major axis length
  b <- sigma_mat[2,2] * sqrt(qchisq(p, 2)) # minor axis length
  phi <- 0 * pi/180 # angle of major axis with x axis phi or tau
  
  t <- seq(0, 2*pi, 0.01) 
  x <- xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi)
  y <- yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi)
  #return(a)
  return(cbind(x, y))
}

sigma_mat <- matrix(c(0.05, 0, 0, 0.05)^2, nrow = 2, ncol = 2)

means_mat <- as.matrix(expand.grid(seq(-4, 4, by = 2), seq(-4, 4, by = 2)))
ell_list <- list()
for(i in 1:nrow(means_mat)){
  ell_list[[i]] <- define_ellipse(means_mat[i, 1], means_mat[i, 2], sqrt(sigma_mat), p = 0.9)
}

# Calculate quality and pMSE for non-private and private data

# Transform back to original scale
tmp_data <- lapply(res_nodp, function(x) trans_back(x, means, sds))

# Calculate percentage of high quality samples
quality_nodp <- sapply(tmp_data, function(data) sum(sapply(ell_list, function(x) 
  min(sum(point.in.polygon(data[, 1], data[, 2], x[, 1], x[, 2])), 1/25*nrow(data))
))/nrow(data))

# Calculate pMSE as defined in the synthpop library
pmse_nodp <- lapply(tmp_data, function(x) gan_pmse(x, data_raw, cp = 0.001))

# Pull out pMSE Ratio score
pmseR_nodp <- sapply(pmse_nodp, function(x) x$utilR)

# Transform back to original scale
tmp_data <- lapply(res_dp, function(x) trans_back(x, means, sds))

# Calculate percentage of high quality samples
quality_dp <- sapply(tmp_data, function(data) sum(sapply(ell_list, function(x) 
  min(sum(point.in.polygon(data[, 1], data[, 2], x[, 1], x[, 2])), 1/25*nrow(data))
))/nrow(data))

# Calculate pMSE as defined in the synthpop library
pmse_dp <- lapply(tmp_data, function(x) gan_pmse(x, data_raw, cp = 0.001))

# Pull out pMSE Ratio score
pmseR_dp <- sapply(pmse_dp, function(x) x$utilR)

res_table <- rbind(pmseR_nodp, pmseR_dp, quality_nodp, quality_dp)

colnames(res_table) <- c("GAN Samples", "DRS Samples", "MWPA Samples", "MWPA+DRS Samples")

rownames(res_table) <- c("pMSE Ratio", "pMSE Ratio DP", "Quality", "Quality DP")

knitr::kable(round(res_table, 3), caption = "Quality of Synthetic Data", format = "latex")

