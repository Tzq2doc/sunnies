source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

### Parameter names
n <- 100  # n: sample size (used 1000 for saved data)
d <- 4    # d: number of features (used 4 for saved data)
N <- 100  # N: number of iterations simulating data (used 1000 for saved data)
data_gen <- dat_unif_squared # data_gen: the data generating function

#### Call shapley_sim1 1000 times with each dependence measure
results <- list()
utilities <- c("R2","DC","BCDC","AIDC","HSIC")
for (i in 1:length(utilities)) {
  results[[utilities[i]]] <- shapley_sim1(
    get(utilities[i]), N, n , d, data_gen)
}
#saveRDS(results, "results/compare_them_all_1000.Rds")
results <- readRDS("results/compare_them_all_1000.Rds")
res_means <- lapply(results, function(r) {apply(r, FUN = mean, MARGIN = 2)})
for (u in utilities) { barplot(res_means[[u]], main = u) }