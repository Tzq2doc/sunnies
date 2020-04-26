source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

# shapley_sim1(1000, 4, 1000, dat_unif_squared)
results <- readRDS("results/sim1_n1000_N1000_d4_unif_squared.Rds")

res_means <- lapply(results, function(r) {apply(r, FUN = mean, MARGIN = 2)})
for (u in utilities) { barplot(res_means[[u]], main = u) }

boxplot(results$R2, outline = F)
boxplot(results$HSIC, outline = F)
boxplot(results$DC, outline = F)
boxplot(results$BCDC, outline = F)
boxplot(results$AIDC, outline = F)












