source("scripts/00-setup.R")
source("../zz-functions.R")
source("../post_GAN_functions.R")

# Define function to generate synthetic data from all four approaches
gen_synth_2010_pums <-
  function(dp = T,
           nmp = 1.5,
           mb = 500,
           l2clip = 1,
           Z_dim = 512,
           n_generators = 50,
           steps = 100,
           MW_epsilon = 0.1,
           last_model = 46780,
           n_samples = 2500,
           window = NULL,
           MW_init = T,
           unweighted_MW = F,
           run_id = 1,
           data = NULL,
           GPU = 0L,
           temp = 0.00001) {
    require(tensorflow)
    require(zeallot)
    require(reticulate)
    
    
    if (dp) {
      model_path <-
        paste0('models/run', run, '/nmp-', nmp, '/pums2010-')
      
    } else {
      model_path <-
        paste0('models/run', run, '/nodp/pums2010-')
    }
    
    tf$reset_default_graph()
    config <- tf$ConfigProto(device_count = list('GPU' = GPU))
    
    
    sess <- tf$Session(config = config)
    
    saver <-
      tf$train$import_meta_graph(paste0(model_path, last_model, ".meta"))
    
    ids <- c(1:last_model)[(1:last_model %% 5 == 0)][]
    length(ids)
    last_ids <- ids[(length(ids) - (n_generators - 1)):length(ids)]
    graph <- tf$get_default_graph()
    
    G_W1 = graph$get_tensor_by_name("G_W1:0")
    G_b1 = graph$get_tensor_by_name("G_b1:0")
    
    G_W1_1 = graph$get_tensor_by_name("G_W1_1:0")
    G_b1_1 = graph$get_tensor_by_name("G_b1_1:0")
    
    G_W1_2 = graph$get_tensor_by_name("G_W1_2:0")
    G_b1_2 = graph$get_tensor_by_name("G_b1_2:0")
    
    G_W2 = graph$get_tensor_by_name("G_W2:0")
    G_b2 = graph$get_tensor_by_name("G_b2:0")
    
    num_sample = tf$constant(as.integer(1), tf$int32)
    
    
    
    generator <- function(Z, temperature = temp) {
      G_h1 = tf$nn$leaky_relu(tf$matmul(Z, G_W1) + G_b1)
      G_h1 = tf$nn$dropout(G_h1, rate = 0.5)
      G_h1_1 = tf$nn$leaky_relu(tf$matmul(G_h1, G_W1_1) + G_b1_1)
      G_h1_1 = tf$nn$dropout(G_h1_1, rate = 0.5)
      G_h1_2 = tf$nn$leaky_relu(tf$matmul(G_h1_1, G_W1_2) + G_b1_2)
      G_h1_2 = tf$nn$dropout(G_h1_2, rate = 0.5)
      G_log_prob = tf$matmul(G_h1_2, G_W2) + G_b2
      
      
      G_age_logits = G_log_prob[, 1, drop = F]
      
      G_sex_logits = tf$concat(list(tf$zeros_like(G_log_prob[, 1, drop = F]), G_log_prob[, 2, drop = F]), 1L)
      G_sex_binary_dist = tf$contrib$distributions$RelaxedOneHotCategorical(temperature, logits = G_sex_logits)
      G_sex_binary = G_sex_binary_dist$sample()
      
      G_hispan_logits = G_log_prob[, 3:27]
      G_hispan_multinom_dist = tf$contrib$distributions$RelaxedOneHotCategorical(temperature, logits = G_hispan_logits)
      G_hispan_multinom = G_hispan_multinom_dist$sample()
      
      G_race_logits = G_log_prob[, 28:38]
      G_race_multinom_dist = tf$contrib$distributions$RelaxedOneHotCategorical(temperature, logits = G_race_logits)
      G_race_multinom = G_race_multinom_dist$sample()
      
      G_puma_logits = G_log_prob[, 39:303]
      G_puma_multinom_dist = tf$contrib$distributions$RelaxedOneHotCategorical(temperature, logits = G_puma_logits)
      G_puma_multinom = G_puma_multinom_dist$sample()
      
      
      G_prob = tf$concat(
        list(
          G_age_logits,
          G_sex_binary[, 2:ncol(G_sex_binary)[[1]], drop = F],
          G_hispan_multinom[, 1:ncol(G_hispan_multinom)[[1]], drop = F],
          G_race_multinom[, 1:ncol(G_race_multinom)[[1]], drop = F],
          G_puma_multinom[, 1:ncol(G_puma_multinom)[[1]], drop = F]
          
        ),
        1L
      )
      
      return(G_prob)
    }
    
    
    Z = tf$placeholder(tf$float32, shape = list(NULL, Z_dim))
    
    
    
    G_sample <- generator(Z)
    sample_Z <- function(m, n) {
      matrix(rnorm(m * n), nrow = m, ncol = n)
    }
    
    X = tf$placeholder(tf$float32, shape = list(NULL, ncol(data)))
    
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
      
      return(D_prob)
    }
    
    D_real = discriminator(X)
    D_fake = discriminator(G_sample)
    
    
    runs <- last_ids
    
    
    total_samples <- n_samples * length(runs)
    
    D_m_real <- NULL
    
    sample_G_m <- NULL
    
    D_m_fake <-
      matrix(NA, nrow = length(runs), ncol = total_samples)
    
    
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
      
      samp_sel <- sample(1:nrow(data), 2000)
      tmp_mean <- mean(sess$run(D_real, feed_dict = dict(X = data[samp_sel,])))
      
      D_m_real <- c(D_m_real, tmp_mean)
      
      D_m_fake[sel, ] <-
        sess$run(D_real, feed_dict = dict(X = sample_G_m))
      cat(which(sel), "\n")
    }
    
    # Sample from last Generator
    
    saver$restore(sess, paste0(model_path, last_model))
    
    sample <-
      sess$run(G_sample, feed_dict = dict(Z = sample_Z(total_samples, Z_dim)))
    
    last_D <-  sess$run(D_real, feed_dict = dict(X = sample))
    
    sample_sel <- rejection_sample(last_D)
    
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


# Get training data
gan_list <- readRDS("gan-input/gan_list.RDS")

data <- as.matrix(gan_list$input_z)

gan_list$input <- NULL
gan_list$input_z <- NULL

# Generate synthetic data from all four approaches
synth_pums <-
  gen_synth_2010_pums(
    last_model = 7445,
    n_generators = 150,
    n_samples = 5000,
    steps = 400,
    run_id = 1,
    data = data,
    unweighted_MW = F,
    MW_epsilon = 0.209,
    window = 150,
    nmp = 1.1,
    MW_init = T
  )


tmp_data <-
  lapply(synth_pums, function(x)
    synth_to_orig(x, gan_list))


# Store synthetic data from all approaches
saveRDS(tmp_data, "synthetic-output/res_df.RDS")