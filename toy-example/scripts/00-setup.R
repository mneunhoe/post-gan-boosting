# Load and potentially install all the packages for this lab

p_needed <-
  c("MASS",
    "tensorflow",
    "zeallot",
    "reticulate",
    "sp",
    "synthpop",
    "magrittr",
    "knitr")
packages <- rownames(installed.packages())
p_to_install <- p_needed[!(p_needed %in% packages)]
if (length(p_to_install) > 0) {
  install.packages(p_to_install)
}
print(sapply(p_needed, require, character.only = TRUE))
#install_tensorflow(version = "1.15.0")
