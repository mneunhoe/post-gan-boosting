source("scripts/00-setup.R")
source("../zz-functions.R")

# Load synthetic samples
runs <- 20

tmp_data_dp <- list()
tmp_data_nodp <- list()
for(i in 1:runs){
tmp <- readRDS(paste0("synthetic-output/res_df_",i,".RDS"))
tmp_data_dp[[i]] <- tmp$dp[[1]]
tmp_data_nodp[[i]] <- tmp$nodp[[1]]
}


# Load real data
gan_list <- readRDS("gan-input/gan_list.RDS")

data <- as.matrix(gan_list$input_z)

gan_list$input <- NULL
gan_list$input_z <- NULL

orig_data <- synth_to_orig(data, gan_list)

# Get test set
test_data <- readRDS("processed-data/test_data.RDS")


# Set formula for model.frame
ff <-
  Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked


set.seed(200611)

# Define function to train random forest models and calculate scores
rf_titanic_results <-
  function(tmp_data,
           runs = 10,
           m = 10,
           test_data = NULL,
           seed = 200131) {
    acc_res <- auc_res <- pr_res <-  array(NA, dim = c(runs, m, 4))
    set.seed(seed)
    for (run in 1:runs) {
      for (i in 1:m) {
        for (model in 1:4) {
          model_df <- tmp_data[[run]][[i]][[model]]
          model_df <- na.omit(model_df)
          more_rows <- nrow(model_df) > nrow(orig_data)
          
          if (more_rows) {
            rf <-
              randomForest(ff, data = model_df[sample(1:nrow(model_df), nrow(orig_data)), ])
          } else {
            rf <- randomForest(ff, data = tmp_data[[run]][[i]][[model]])
          }
          predictions <- predict(rf, newdata = test_data)
          res_real <-
            table(predictions,
                  test_data$Survived)
          
          acc_res[run, i, model] <-
            sum(diag(res_real)) / sum(res_real)
          
          predictions <-
            predict(rf, newdata = test_data, type = "prob")
          auc_res[run, i, model] <-
            roc.curve(
              scores.class0 = predictions[, 2],
              weights.class0 = as.numeric(test_data$Survived) - 1
            )$auc
          
          
          pr_res[run, i, model] <-
            pr.curve(
              scores.class0 = predictions[, 2],
              weights.class0 = as.numeric(test_data$Survived) - 1
            )$auc.integral
          
          print(res_real)
        }
      }
    }
    acc_res <-
      provideDimnames(acc_res ,
                      sep = "_",
                      base = list('run', 'm', 'model'))
    
    auc_res <-
      provideDimnames(auc_res ,
                      sep = "_",
                      base = list('run', 'm', 'model'))
    
    pr_res <-
      provideDimnames(pr_res ,
                      sep = "_",
                      base = list('run', 'm', 'model'))
    return(list(acc = acc_res, auc = auc_res, pr = pr_res))
  }

rf_res_nodp <-
  rf_titanic_results(
    tmp_data = tmp_data_nodp,
    runs = runs,
    m = m,
    test_data = test_data
  )

rf_res_dp <-
  rf_titanic_results(
    tmp_data = tmp_data_dp,
    runs = runs,
    m = m,
    test_data = test_data
  )


# Define function to train logit models and calculate scores
logit_titanic_results <-
  function(tmp_data,
           runs = 10,
           m = 10,
           test_data = NULL,
           seed = 200131) {
    acc_res <- auc_res <- pr_res <-  array(NA, dim = c(runs, m, 4))
    set.seed(seed)
    for (run in 1:runs) {
      for (i in 1:m) {
        for (model in 1:4) {
          model_df <- tmp_data[[run]][[i]][[model]]
          model_df <- na.omit(model_df)
          more_rows <- nrow(model_df) > nrow(orig_data)
          
          if (more_rows) {
            rf <-
              glm(ff,
                  data = model_df[sample(1:nrow(model_df), nrow(orig_data)), ],
                  family = binomial(link = "logit"))
          } else {
            rf <- glm(ff,
                      data = model_df,
                      family = binomial(link = "logit"))
          }
          
          
          predictions <-
            predict(rf,
                    newdata = remove_missing_levels(rf, test_data),
                    type = "response")
          sel <- !is.na(predictions)
          res_real <-
            table(predictions > 0.5,
                  test_data$Survived)
          
          acc_res[run, i, model] <-
            sum(diag(res_real)) / sum(res_real)
          
          auc_res[run, i, model] <-
            roc.curve(
              scores.class0 = predictions[sel],
              weights.class0 = as.numeric(test_data$Survived[sel]) - 1
            )$auc
          
          
          pr_res[run, i, model] <-
            pr.curve(
              scores.class0 = predictions[sel],
              weights.class0 = as.numeric(test_data$Survived[sel]) - 1
            )$auc.integral
          
          print(res_real)
        }
      }
    }
    acc_res <-
      provideDimnames(acc_res ,
                      sep = "_",
                      base = list('run', 'm', 'model'))
    
    auc_res <-
      provideDimnames(auc_res ,
                      sep = "_",
                      base = list('run', 'm', 'model'))
    
    pr_res <-
      provideDimnames(pr_res ,
                      sep = "_",
                      base = list('run', 'm', 'model'))
    return(list(acc = acc_res, auc = auc_res, pr = pr_res))
  }


logit_res_nodp <-
  logit_titanic_results(
    tmp_data = tmp_data_nodp,
    runs = runs,
    m = m,
    test_data = test_data
  )


logit_res_dp <-
  logit_titanic_results(
    tmp_data = tmp_data_dp,
    runs = runs,
    m = m,
    test_data = test_data
  )

# Define function to train xgb models and calculate scores
xgb_titanic_results <-
  function(tmp_data,
           runs = 10,
           m = 10,
           test_data = NULL,
           seed = 200131) {
    acc_res <- auc_res <- pr_res <-  array(NA, dim = c(runs, m, 4))
    set.seed(seed)
    for (run in 1:runs) {
      for (i in 1:m) {
        for (model in 1:4) {
          model_df <- tmp_data[[run]][[i]][[model]]
          model_df <- na.omit(model_df)
          more_rows <- nrow(model_df) > nrow(orig_data)
          mf_train <- model.frame(ff, model_df)
          mm_train <- model.matrix(ff, mf_train)
          
          
          if (more_rows) {
            sel <- sample(1:nrow(model_df), nrow(orig_data))
            rf <-
              xgboost(
                data = mm_train[sel, 2:ncol(mf_train)],
                label = as.numeric(mf_train[sel, 1]) - 1,
                verbose = 1,
                max.depth = 6,
                eta = 0.3,
                nthread = 6,
                nrounds = 10,
                objective = "binary:logistic"
              )
          } else {
            rf <-
              xgboost(
                data = mm_train[, 2:ncol(mf_train)],
                label = as.numeric(mf_train[, 1]) - 1,
                verbose = 1,
                max.depth = 6,
                eta = 0.3,
                nthread = 6,
                nrounds = 25,
                objective = "binary:logistic"
              )
          }
          
          mf_test <- model.frame(ff, test_data)
          mm_test <- model.matrix(ff, mf_test)
          
          predictions <-
            predict(rf, type = "prob", newdata = mm_test[, 2:ncol(mf_test)])
          sel <- !is.na(predictions)
          res_real <-
            table(predictions > 0.5,
                  test_data$Survived)
          
          acc_res[run, i, model] <-
            sum(diag(res_real)) / sum(res_real)
          
          auc_res[run, i, model] <-
            roc.curve(
              scores.class0 = predictions[sel],
              weights.class0 = as.numeric(test_data$Survived[sel]) - 1
            )$auc
          
          
          pr_res[run, i, model] <-
            pr.curve(
              scores.class0 = predictions[sel],
              weights.class0 = as.numeric(test_data$Survived[sel]) - 1
            )$auc.integral
          
          print(res_real)
        }
      }
    }
    acc_res <-
      provideDimnames(acc_res ,
                      sep = "_",
                      base = list('run', 'm', 'model'))
    
    auc_res <-
      provideDimnames(auc_res ,
                      sep = "_",
                      base = list('run', 'm', 'model'))
    
    pr_res <-
      provideDimnames(pr_res ,
                      sep = "_",
                      base = list('run', 'm', 'model'))
    return(list(acc = acc_res, auc = auc_res, pr = pr_res))
  }

xgb_res_nodp <-
  xgb_titanic_results(
    tmp_data = tmp_data_nodp,
    runs = runs,
    m = m,
    test_data = test_data
  )

xgb_res_dp <-
  xgb_titanic_results(
    tmp_data = tmp_data_dp,
    runs = runs,
    m = m,
    test_data = test_data
  )

# Summarize results in one table
res <- rbind(t(sapply(logit_res_nodp, function(x)
  apply(x, 3, median))),
  t(sapply(rf_res_nodp, function(x)
    apply(x, 3, median))),
  t(sapply(xgb_res_nodp, function(x)
    apply(x, 3, median))),
  
  t(sapply(logit_res_dp, function(x)
    apply(x, 3, median))),
  t(sapply(rf_res_dp, function(x)
    apply(x, 3, median))),
  t(sapply(xgb_res_dp, function(x)
    apply(x, 3, median))))


colnames(res) <- c("GAN", "DRS", "PGB", "PGB+DRS")
rownames(res) <- rep(c("Accuracy", "ROC AUC", "PR AUC"), 6)
knitr::kable(round(res, 3), "latex", booktabs = T)


res_sd <- rbind(t(sapply(logit_res_nodp, function(x)
  apply(x, 3, sd))),
  t(sapply(rf_res_nodp, function(x)
    apply(x, 3, sd))),
  t(sapply(xgb_res_nodp, function(x)
    apply(x, 3, sd))),
  
  t(sapply(logit_res_dp, function(x)
    apply(x, 3, sd))),
  t(sapply(rf_res_dp, function(x)
    apply(x, 3, sd))),
  t(sapply(xgb_res_dp, function(x)
    apply(x, 3, sd))))


colnames(res_sd) <- c("GAN", "DRS", "PGB", "PGB+DRS")
rownames(res_sd) <- rep(c("Accuracy", "ROC AUC", "PR AUC"), 6)
knitr::kable(round(res_sd, 3), "latex", booktabs = T)


tmp_data_real <- list(list(list(orig_data,
                                orig_data,
                                orig_data,
                                orig_data)))

xgb_res_real <-
  xgb_titanic_results(
    tmp_data = tmp_data_real,
    runs = 1,
    m = 1,
    test_data = test_data
  )
rf_res_real <-
  rf_titanic_results(
    tmp_data = tmp_data_real,
    runs = 1,
    m = 1,
    test_data = test_data
  )

logit_res_real <-
  logit_titanic_results(
    tmp_data = tmp_data_real,
    runs = 1,
    m = 1,
    test_data = test_data
  )


real_res <- rbind(t(sapply(logit_res_real, function(x)
  apply(x, 3, mean))),
  t(sapply(rf_res_real, function(x)
    apply(x, 3, mean))),
  t(sapply(xgb_res_real, function(x)
    apply(x, 3, mean))))[,1]

knitr::kable(cbind(names(real_res), (round(real_res, 3))), "latex", booktabs = T)

res
