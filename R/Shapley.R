source("old/old_shapley_helpers.R")
source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

results <- readRDS("results/compare_them_all_1000.Rds")

boxplot(results$R2, outline = F)
boxplot(results$HSIC, outline = F)
boxplot(results$DC, outline = F)
boxplot(results$BCDC, outline = F)
boxplot(results$AIDC, outline = F)


# y = 2*0*x_1^2 + 2*1*x_2^2 + 2*2*x_3^2 + ... + 2*(d-1)*x_{d-1}^2
plot(X[,1], y)
points(X[,2], y, col = "green")
points(X[,3], y, col = "blue")
points(X[,4], y, col = "red")

barplot(results[1,])
barplot(results[2,])









