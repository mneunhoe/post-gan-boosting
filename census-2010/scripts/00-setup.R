# Load and potentially install all the packages for this project

p_needed <-
  c("MASS",
    "tensorflow",
    "zeallot",
    "reticulate",
    "sp",
    "synthpop",
    "magrittr",
    "knitr",
    "ipumsr",
    "data.table")
packages <- rownames(installed.packages())
p_to_install <- p_needed[!(p_needed %in% packages)]
if (length(p_to_install) > 0) {
  install.packages(p_to_install)
}
print(sapply(p_needed, require, character.only = TRUE))
