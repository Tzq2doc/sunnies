source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

loc1 <- shapley_sim1_N(1000, 4, 1000, dat_unif_squared, loc_only = T)
loc2 <- shapley_sim1_N(100, 4, 100, dat_unif_squared, loc_only = T)

#results <- readRDS(loc1)
results <- readRDS(loc2)

res_means <- lapply(results, function(r) {apply(r, FUN = mean, MARGIN = 2)})
for (u in utilities) { barplot(res_means[[u]], main = u) }

boxplot(results$R2, outline = F)
boxplot(results$HSIC, outline = F)
boxplot(results$DC, outline = F)
boxplot(results$BCDC, outline = F)
boxplot(results$AIDC, outline = F)








