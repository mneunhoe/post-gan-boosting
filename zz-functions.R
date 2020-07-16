

gan_input <- function(df, type, contrasts = F, means = NULL, sds = NULL) {
  if (ncol(df) != length(type)) {
    stop("A type needs to be specified for every column.")
  }
  df_ord <-
    cbind(df[, type == "num", drop = F], df[, type == "bin", drop = F], df[, type == "fac", drop = F])
  reorder <- match(colnames(df), colnames(df_ord))
  order_type <-  match(colnames(df_ord), colnames(df))
  type_order <- type[order_type]
  cat <- which(type_order == "fac")
  if (length(cat) != 0) {
    conts <- list()
    levs <- list()
    
    for (i in 1:length(cat)) {
      levs[[i]] <- levels(df_ord[, cat[i]])
      conts[[i]] <-
        as.matrix(contrasts(df_ord[, cat[i]], contrasts = contrasts)[as.numeric(df_ord[, cat[i]]), ], nrow = nrow(df_ord))
      colnames(conts[[i]]) <-
        paste(colnames(df_ord)[cat[i]], colnames(conts[[i]]), sep = ".")
    }
    cont <- do.call(cbind, conts)
    out <- cbind(df_ord[, 1:(min(cat) - 1)], cont)
  } else{
    levs <- NULL
    out <- df_ord
  }
  out <- as.matrix(out)
  rownames(out) <- rownames(df_ord)
  
  
  out_z <- out
  
  num_out <- out[, which(type_order == "num"), drop = F]
  
  
  if(is.null(means)){
    means <- apply(num_out, 2, mean, na.rm = T)
    sds <- apply(num_out, 2, sd, na.rm = T)
  }
  
  num_z <- as.matrix(sweep(sweep(num_out, 2, means), 2, sds, "/"))
  
  out_z[, which(type_order == "num")] <- num_z
  
  output <-  list(
    input = as.matrix(out),
    input_z = as.matrix(out_z),
    input_names = colnames(df_ord),
    reorder = reorder,
    type_order = type_order,
    levels = levs,
    means = means,
    sds = sds
  )
  
  class(output) <- "gan_input"
  
  return(
    output
  )
}

synth_to_orig <- function(synth_data, rgain) {
  if (sum(rgain$type_order != "num") > 0) {
    synth_data[, (sum(rgain$type_order == "num") + 1):ncol(synth_data)] <-
      round(synth_data[, (sum(rgain$type_order == "num") + 1):ncol(synth_data)], 0)
    
    synth_data[, (sum(rgain$type_order == "num") + 1):ncol(synth_data)][synth_data[, (sum(rgain$type_order == "num") + 1):ncol(synth_data)]>1] <- 1
    synth_data[, (sum(rgain$type_order == "num") + 1):ncol(synth_data)][synth_data[, (sum(rgain$type_order == "num") + 1):ncol(synth_data)]<0] <- 0
    
    }
  
  if (length(1:sum(rgain$type_order == "num")) == 1) {
    synth_data[, 1] <-
      sweep(sweep(synth_data[, 1:sum(rgain$type_order == "num"), drop = F] , 2, rgain$sds, "*"), 2, rgain$means, "+")
  } else {
    synth_data[, 1:sum(rgain$type_order == "num")] <-
      sweep(sweep(synth_data[, 1:sum(rgain$type_order == "num"), drop = F] , 2, rgain$sds, "*"), 2, rgain$means, "+")
  }
  rgain_synth <- rgain
  rgain_synth$input <- synth_data
  
  
  cat <- which(rgain_synth$type_order == "fac")
  if (length(cat) != 0) {
    facs <- list()
    i <- 1
    for (i in 1:length(cat)) {
      sel <-
        grep(rgain_synth$input_names[rgain_synth$type_order == "fac"][i],
             colnames(rgain_synth$input))
      dummies <- rgain_synth$input[, sel, drop = F]
      facs[[i]] <-
        factor(
          dummies %*% 1:ncol(dummies),
          labels = rgain_synth$levels[[i]] ,
          levels = 1:length(rgain_synth$levels[[i]])
        )
    } # What if we have missing categories in between?
    
    out <-
      data.frame(rgain_synth$input[, 1:(min(cat) - 1)], do.call(data.frame, facs))
  } else{
    out <- rgain_synth$input
  }
  colnames(out) <- rgain_synth$input_names
  out <- out[, rgain_synth$reorder]
  return(out)
}

gan_pmse <- function(x,
                     true_data,
                     method = "cart",
                     cp = 0.05,
                     n = 3000, seed = 200605,
                     ...) {
  set.seed(seed)
  n_true <- nrow(true_data)
  n_synth <- nrow(x)
  if (n_synth >= n) {
    a <- data.frame(true_data[sample(1:n_true, n),])
    b <- data.frame(x[sample(1:n_synth, n),])
    colnames(b) <- colnames(a)
  }
  else if (n_true >= n_synth) {
    a <- data.frame(true_data[sample(1:n_true, n_synth), ])
    b <- data.frame(x)
    colnames(b) <- colnames(a)
  } else {
    a <- data.frame(true_data)
    b <- data.frame(x[sample(1:n_synth, n_true), ])
    colnames(b) <- colnames(a)
  }
  syn_x <- list(m = 1, syn = b)
  require(synthpop)
  class(syn_x) <- "synds"
  ut <- utility.gen(syn_x, a, method = method, cp = cp, ...)
  return(ut)
}

create_bp_mat <- function(orig_data, samples, variable = "race"){
  bp_mat <- cbind(table(orig_data[,variable])/sum(table(orig_data[,variable])), sapply(samples, function(run) lapply(run, function(m) sapply(m, function(x) table(x[,variable])/sum(table(x[,variable])) )))[[1]])
  colnames(bp_mat) <- c("Real Data", "GAN", "DRS", "PBG", "PBG+DRS")
  
  return(bp_mat)
}

library(magrittr)

#' @title remove_missing_levels
#' @description Accounts for missing factor levels present only in test data
#' but not in train data by setting values to NA
#'
#' @import magrittr
#' @importFrom gdata unmatrix
#' @importFrom stringr str_split
#'
#' @param fit fitted model on training data
#'
#' @param test_data data to make predictions for
#'
#' @return data.frame with matching factor levels to fitted model
#'
#' @keywords internal
#'
#' @export
remove_missing_levels <- function(fit, test_data) {
  
  # https://stackoverflow.com/a/39495480/4185785
  
  # drop empty factor levels in test data
  test_data %>%
    droplevels() %>%
    as.data.frame() -> test_data
  
  # 'fit' object structure of 'lm' and 'glmmPQL' is different so we need to
  # account for it
  if (any(class(fit) == "glmmPQL")) {
    # Obtain factor predictors in the model and their levels
    factors <- (gsub("[-^0-9]|as.factor|\\(|\\)", "",
                     names(unlist(fit$contrasts))))
    # do nothing if no factors are present
    if (length(factors) == 0) {
      return(test_data)
    }
    
    map(fit$contrasts, function(x) names(unmatrix(x))) %>%
      unlist() -> factor_levels
    factor_levels %>% str_split(":", simplify = TRUE) %>%
      extract(, 1) -> factor_levels
    
    model_factors <- as.data.frame(cbind(factors, factor_levels))
  } else {
    # Obtain factor predictors in the model and their levels
    factors <- (gsub("[-^0-9]|as.factor|\\(|\\)", "",
                     names(unlist(fit$xlevels))))
    # do nothing if no factors are present
    if (length(factors) == 0) {
      return(test_data)
    }
    
    factor_levels <- unname(unlist(fit$xlevels))
    model_factors <- as.data.frame(cbind(factors, factor_levels))
  }
  
  # Select column names in test data that are factor predictors in
  # trained model
  
  predictors <- names(test_data[names(test_data) %in% factors])
  
  # For each factor predictor in your data, if the level is not in the model,
  # set the value to NA
  
  for (i in 1:length(predictors)) {
    found <- test_data[, predictors[i]] %in% model_factors[
      model_factors$factors == predictors[i], ]$factor_levels
    if (any(!found)) {
      # track which variable
      var <- predictors[i]
      # set to NA
      test_data[!found, predictors[i]] <- NA
      # drop empty factor levels in test data
      test_data %>%
        droplevels() -> test_data
      # issue warning to console
      message(sprintf(paste0("Setting missing levels in '%s', only present",
                             " in test data but missing in train data,",
                             " to 'NA'."),
                      var))
    }
  }
  return(test_data)
}


rmse <- function(pred, obs) {
  sqrt(mean((pred - obs) ^ 2))
}