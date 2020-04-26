source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

### Parameter names
# n: sample size (used 1000 for saved data)
# d: number of features (used 4 for saved data)
# N: number of iterations simulating data (used 1000 for saved data)
# data_gen: the data generating function
#### Call shapley_sim1 1000 times with each dependence measure
shapley_sim1 <- function(n, d, N, data_gen, overwrite = F) {
  results <- list()
  utilities <- c("R2","DC","BCDC","AIDC","HSIC")
  for (i in 1:length(utilities)) {
    results[[utilities[i]]] <- shapley_sim1(
      get(utilities[i]), N, n , d, data_gen)
  }
  fun_name <- as.character(substitute(data_gen))
  dir <- "results/"
  loc <- paste0(dir, "sim1_n",n,"_N",N,"_d",d,"_", fun_name,".Rds")
  if (!overwrite && file.exists(loc)) {
    stop("file exists, perhaps choose overwrite = T")}
  saveRDS(results, loc)
}