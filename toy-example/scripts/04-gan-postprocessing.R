source("scripts/00-setup.R")
source("../post_GAN_functions.R")

data <-
  read.csv(
    "gan-input/gaussian_df_z.csv",
    header = T
  )
data <- as.matrix(data)


# Define function that generates data from all four approaches
# directly from the stored weights

gen_synth_25_gaussians <-
  function(dp = T,
           nmp = 2.165,
           mb = 50,
           l2clip = 1,
           Z_dim = 25,
           n_generators = 50,
           steps = 100,
           MW_epsilon = 0.1,
           last_model = 5000,
           n_samples = 2500,
           window = NULL,
           MW_init = F,
           run = 1,
           GPU = 0L,
           data) {
    require(tensorflow)
    require(zeallot)
    require(reticulate)
    
    if (dp) {
      model_path <-
        paste0("models/run", run, "/nmp-", nmp, "-mb-", mb, "-l2clip-", l2clip, "/toy-example-")
    } else {
      model_path <- paste0("models/run",run,"/nodp/toy-example-")
    }
    
    tf$reset_default_graph()
    config <- tf$ConfigProto(device_count = list('GPU' = GPU))
    
    sess <- tf$Session(config = config)
    
    # Load meta graph and restore weights
    saver <-
      tf$train$import_meta_graph(paste0(model_path, last_model, ".meta"))
    
    ids <- c(1:last_model)[(1:last_model %% 5 == 0)][]

    last_ids <- ids[(length(ids) - (n_generators - 1)):length(ids)]
    graph <- tf$get_default_graph()
    
    # Recreate Generator weights and Generator
    G_W1 = graph$get_tensor_by_name("G_W1:0")
    G_b1 = graph$get_tensor_by_name("G_b1:0")
    
    G_W1_1 = graph$get_tensor_by_name("G_W1_1:0")
    G_b1_1 = graph$get_tensor_by_name("G_b1_1:0")
    
    G_W1_2 = graph$get_tensor_by_name("G_W1_2:0")
    G_b1_2 = graph$get_tensor_by_name("G_b1_2:0")
    
    G_W2 = graph$get_tensor_by_name("G_W2:0")
    G_b2 = graph$get_tensor_by_name("G_b2:0")
    
    generator <- function(z) {
      G_h1 = tf$nn$leaky_relu(tf$matmul(Z, G_W1) + G_b1)
      G_h1 = tf$nn$dropout(G_h1, rate = 0.5)
      
      G_h1_1 = tf$nn$leaky_relu(tf$matmul(G_h1, G_W1_1) + G_b1_1)
      G_h1_1 = tf$nn$dropout(G_h1_1, rate = 0.5)
      
      G_h1_2 = tf$nn$leaky_relu(tf$matmul(G_h1_1, G_W1_2) + G_b1_2)
      G_h1_2 = tf$nn$dropout(G_h1_2, rate = 0.5)
      
      G_log_prob = tf$matmul(G_h1_2, G_W2) + G_b2
      
      
      G_prob = G_log_prob
      
      return(G_prob)
    }
    
    
    Z = tf$placeholder(tf$float32, shape = list(NULL, Z_dim))
    
    G_sample <- generator(Z)
    
    sample_Z <- function(m, n) {
      matrix(rnorm(m * n), nrow = m, ncol = n)
    }
    
    X = tf$placeholder(tf$float32, shape = list(NULL, ncol(data)))
    
    # Recreate Discriminator weights and Discriminator
    
    D_W1 = graph$get_tensor_by_name("D_W1:0")
    D_b1 = graph$get_tensor_by_name("D_b1:0")
    
    D_W1_1 = graph$get_tensor_by_name("D_W1_1:0")
    D_b1_1 = graph$get_tensor_by_name("D_b1_1:0")
    
    D_W1_2 = graph$get_tensor_by_name("D_W1_2:0")
    D_b1_2 = graph$get_tensor_by_name("D_b1_2:0")
    
    D_W2 = graph$get_tensor_by_name("D_W2:0")
    D_b2 = graph$get_tensor_by_name("D_b2:0")
    
    discriminator <- function(x) {
      D_h1 = tf$nn$leaky_relu(tf$matmul(x, D_W1) + D_b1)
      
      D_h1_1 = tf$nn$leaky_relu(tf$matmul(D_h1, D_W1_1) + D_b1_1)
      
      D_h1_2 = tf$nn$leaky_relu(tf$matmul(D_h1_1, D_W1_2) + D_b1_2)
      
      D_logit = tf$matmul(D_h1_2, D_W2) + D_b2
      D_prob = tf$nn$sigmoid(D_logit)
      
      return(list(D_prob, D_logit))
    }
    
    c(D_real, D_logit_real) %<-% discriminator(X)
    c(D_fake, D_logit_fake) %<-% discriminator(G_sample)
    
    
    D_loss_real <-
      tf$nn$sigmoid_cross_entropy_with_logits(logits = D_logit_real,
                                              labels = tf$ones_like(D_logit_real))
    D_loss_fake <-
      tf$nn$sigmoid_cross_entropy_with_logits(logits = D_logit_fake,
                                              labels = tf$ones_like(D_logit_fake))
    
    vector_D_loss <- D_loss_real + D_loss_fake
    D_loss <- tf$reduce_mean(vector_D_loss)
    
    
    
    
    # Now create samples and from the last generators
    runs <- last_ids
    
    # Number of total samples drawn across all generators
    total_samples <- n_samples * length(runs)
    
    # Empty matrix to hold the discriminator scores of fake data from each discriminator
    D_m_fake <-
      matrix(NA, nrow = length(runs), ncol = total_samples)
    
    # Empty object to get discriminator scores on real data from each discriminator
    D_m_real <- NULL
    
    # Empty object to hold the samples from each generator (B in the paper)
    sample_G_m <- NULL
    
   
    # Loop over last runs and append sample_G_m = B
    for (i in runs) {
      
      saver$restore(sess, paste0(model_path, i))
      
      sample_G_m <-
        rbind(sample_G_m, sess$run(G_sample, feed_dict = dict(Z = sample_Z(
          n_samples, Z_dim
        ))))
    }
    
    for (i in runs) {
      sel <- runs == i
      saver$restore(sess, paste0(model_path, i))
      
      tmp_mean <- mean(sess$run(D_real, feed_dict = dict(X = data)))
      
      D_m_real <- c(D_m_real, tmp_mean)
      
      D_m_fake[sel, ] <-
        sess$run(D_real, feed_dict = dict(X = sample_G_m))
      cat(which(sel), "\n")
    }
    
    # Sample from last Generator (for comparison)
    # Sample from last Generator
    saver$restore(sess, paste0(model_path, last_model))
    
    # Sample 50,000 examples from last Generator for DRS to work
    sample <-
      sess$run(G_sample, feed_dict = dict(Z = sample_Z(50000, Z_dim)))
    
    # Get Discriminator scores of sample for DRS
    last_D <-  sess$run(D_real, feed_dict = dict(X = sample))
    
    # Get DRS sample indices
    sample_sel <- rejection_sample(last_D)
    # Subset last Generator sample to DRS sample
    o_DRS <- sample[sample_sel, ]
    
    # Get post-GAN Boosting sample and Discriminator scores
    c(sample_PGB, D_PGB) %<-% post_gan_boosting(d_score_fake = D_m_fake,
                               d_score_real = D_m_real,
                               B = sample_G_m,
                               real_N = nrow(data),
                               steps = steps,
                               MW_epsilon = MW_epsilon,
                               N_generators = length(runs),
                               averaging_window = window)
    
    shuf_sel <- sample(length(D_PGB), length(D_PGB))
    
    D_PGB_shuf <- D_PGB[shuf_sel]
    
    sample_sel <- rejection_sample(D_PGB)
    sample_PGB_DRS <- sample_PGB[sample_sel, ]
    
    
    colnames(sample_PGB) <- colnames(data)
    colnames(sample_PGB_DRS) <- colnames(data)
    colnames(sample) <- colnames(data)
    colnames(o_DRS) <- colnames(data)
    
    sess$close()
    
    return(
      list(
        sample = sample,
        o_DRS = o_DRS,
        sample_MW = sample_PGB,
        sample_DRS = sample_PGB_DRS
      )
    )
    
  }


# Set number of PGB steps
steps <- 400

# Get results with Privacy
res_dp <-
  gen_synth_25_gaussians(
    dp = T,
    nmp = 1.5,
    mb = 50,
    l2clip = 1,
    Z_dim = 32,
    n_generators = 200,
    steps = steps,
    MW_epsilon = 0.269,
    last_model = 10000,
    n_samples = 2000,
    window = NULL,
    MW_init = F,
    run = 1,
    data = data
  )

# Save results for summarizing later
saveRDS(res_dp, "synthetic-output/res_dp.RDS")

# Get results without privacy
res_nodp <-
  gen_synth_25_gaussians(
    dp = F,
    Z_dim = 32,
    n_generators = 200,
    steps = steps,
    last_model = 10000,
    n_samples = 2000,
    window = NULL,
    MW_init = F,
    run = 1,
    data = data
  )

# Save results for summarizing
saveRDS(res_nodp, "synthetic-output/res_nodp.RDS")
