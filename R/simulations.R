source("datasets/simulated_datasets.R")
source("simulation_helpers.R")



# Singleton simulations ---------------------------------------------------

singletons_sim_N(n=1000, d=4, N=1000, dat_unif_squared)




# Shapley simulations sim1 ------------------------------------------------

loc1 <- shapley_sim1_N(n=1000, d=4, N=1000, dat_unif_squared, loc_only = T)
loc2 <- shapley_sim1_N(n=100,  d=4, N=100,  dat_unif_squared, loc_only = T)
loc3 <- shapley_sim1_N(n=10,   d=4, N=100,  dat_unif_squared, loc_only = T)
loc4 <- shapley_sim1_N(n=100,  d=5, N=100,  dat_unif_squared, loc_only = T)
loc5 <- shapley_sim1_N(n=100,  d=3, N=100,  dat_unif_squared)


# feature means of simulations results ------------------------------------

#results <- readRDS(loc1)
results <- readRDS(loc4)

res_means <- lapply(results, function(r) {apply(r, FUN = mean, MARGIN = 2)})
for (u in utilities) { barplot(res_means[[u]], main = u) }


# Boxplots of simulation results ------------------------------------------

boxplot(results$R2, outline = F)
boxplot(results$HSIC, outline = F)
boxplot(results$DC, outline = F)
boxplot(results$BCDC, outline = F)
boxplot(results$AIDC, outline = F)

normalise_res <- function(x) {(x - mean(x))/sd(x)}
results$HSIC <- normalise_res(results$HSIC)
results$DC <- normalise_res(results$DC)