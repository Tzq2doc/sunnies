source("datasets/simulated_datasets.R")
source("shapley_helpers.R")
source("utility_functions.R")
source("comparison_helpers.R")
source("shapley_helpers.R")
source("Applications_helpers.R")
library(xgboost)
library(dplyr)
library(tidyr)
library(ggplot2)
library(latex2exp)
library(reticulate)
library(reshape2)

### ERDA Example 1 -----------------------------------------------------------
# The repetitions producing violin plots is in python, but here is similar:
result1 <- run_evaluations(dat_unif_squared, utility = DC, n = 1e3, plots = T)

### ERDA Examples 2 and 3  ----------------------------------------------------
# (which should be the same example)
# This example should be done exactly by hand, but here is a simulation.
# When it is calculated exactly, the shapley values will be equal to each other:
result2 <- run_evaluations(dat_catcat_XOR, utility = DC, n = 1e3, plots = T)

### DARRP EXAMPLE 1 -----------------------------------------------------------
# One feature becomes more important, one less important.
n <- 1e4; m <- 10; d <- 4; N <- 100; s <- 1000
shaps_lab <- matrix(0, nrow = m+1, ncol = d) 
shaps_res <- matrix(0, nrow = m+1, ncol = d)
mse <- vector(mode = "numeric", length = m+1)
datt <- dat_t(n, d, t = 0, max_t = m)
sdatt <- split_dat(datt)
xgb <- basic_xgb_fit(sdatt) # Model is only fit once
cat("Number of features: ", xgb$nfeatures)
cdN_all <- array(NA, dim = c(6,d,N,m))
for (t in 0:m) {
  print(paste0("round ",t))
  xgbtt <- basic_xgb_test(xgb, sdatt) # Model is tested each time
  mse[t+1] <- xgbtt$test_mse
  cdNt <- compare_DARRP_N(sdatt, xgbtt, features = 1:4, 
                         feature_names = paste0("x",1:4),
                         sample_size = s, N = N,
                         valid = F, all_labs = F)
  cdN_all[,,,t] <- cdNt
  datt <- dat_t(n, d, t = t, max_t = m)
  sdatt <- split_dat(datt)
}
#saveRDS(cdN_all, "run1_cdN_drift.Rds")
#saveRDS(mse, "run1_cdN_drift_mse.Rds")
cdN_all <- readRDS("results/run1_cdN_drift.Rds")
mse <- readRDS("results/run1_cdN_drift_mse.Rds")
pdf(file="Drift_ADL_ADR.pdf",width=5,height=4)
plot_compare_DARRP_N_drift(cdN_all, shap_index = c(1,5))
dev.off()
# ------------
pdf(file="Drift_ADL_ADR_lines.pdf",width=5,height=4)
plot_compare_DARRP_N_drift2(cdN_all, shap_index = c(1,5))
dev.off()


### DARRP EXAMPLE 2 -----------------------------------------------------------
n <- 1e4; d <- 5
X <- matrix(rnorm(n*(d-2),0,1), nrow = n, ncol = (d-2))
X4 <- sample(0:1, replace = T, n)
X5 <- sample(0:1, replace = T, n)
y <- rowSums(X[,-(d-2)]) + 5*(X4 & X5)*X[,(d-2)] + rnorm(n,0,0.1)
dat <- cbind(y,X,X4,X5)
sdat <- split_dat(dat, df = T)
lmodel <- lm(y ~ x1 + x2 + x3 + x4 + x5, dat = sdat$df_yx_train)
lmodelt <- basic_lmodel_test(lmodel, sdat)
cdN <- compare_DARRP_N(sdat, lmodelt, features = 1:5, 
                       feature_names = paste0("x",1:5),
                       sample_size = 1000, N = 100,
                       valid = F, all_labs = F)
#saveRDS(cdN, "run1_cdN_linear1.Rds")
cdN <- readRDS("results/run1_cdN_linear1.Rds")
plot_compare_DARRP_N(cdN, main = "test run", all_labs = F)
lmodel2 <- lm(y ~ x1 + x2 + x3 + x4 + x5 + x3:x4:x5, dat = sdat$df_yx_train)
lmodel2t <- basic_lmodel_test(lmodel2, sdat)
cdN2 <- compare_DARRP_N(sdat, lmodel2t, features = 1:5, 
                       feature_names = paste0("x",1:5),
                       sample_size = 1000, N = 100,
                       valid = F, all_labs = F)
#saveRDS(cdN2, "run1_cdN_linear2.Rds")
plot_compare_DARRP_N(cdN2, main = "test run", all_labs = F)

plot_compare_DARRP_N_interact <- function(
  cdN_all, p = c(0.025,0.975),
  d = dim(cdN_all)[2],
  feature_names = paste0("X",1:d),
  y_name = "Shapley value",
  leg_labs = unname(TeX(paste0("$X_",1:d)))) {
  
  s <- dim(cdN_all)[1]
  d <- dim(cdN_all)[2]
  N <- dim(cdN_all)[3]
  cd2 <- matrix(ncol = s, nrow = 0)
  cdN <- matrix(ncol = d, nrow = 0)
  dimnames(cdN_all) <- list(
      S = paste0("S",1:s), feature = feature_names, i = 1:N)
  cd <- cdN_all %>% 
    apply(MARGIN = c(1,2), FUN = function(x){ 
        c(mean(x), quantile(x, probs = p)) 
    })
  for (i in 1:d) {cd2 <- rbind(cd2, cd[,,i])}
  for (i in 1:N) {cdN <- rbind(cdN, cdN_all[,,i])}
  cdN <- cdN %>% 
    data.frame() %>%
    cbind(i = rep(1:N,each=s), S = paste0("S",1:s), .)
    
  cd2 <- cd2 %>% 
    data.frame() %>% 
    cbind(feature = rep(feature_names,each=3), CI = 1:3) %>% 
    pivot_longer(all_of(paste0("S",1:s)), names_to = "S")
  
  cd2 <- cd2 %>% mutate(type=recode(S, S1="ADL", S3="ADP")) %>% 
    select(-S)
  cd3 <- cd2 %>% 
    filter(type %in% c("ADL","ADP") & CI == 1) %>% 
    pivot_wider(names_from = type, values_from = value)
  cdCI <- cd2 %>% 
    filter(type %in% c("ADL","ADP")) %>% 
    pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
  cdN2 <- cdN %>% 
    mutate(type=recode(S, S1="ADL", S3="ADP")) %>% 
    select(-S) %>% 
    filter(type %in% c("ADL","ADP")) %>% 
    pivot_longer(all_of(feature_names), names_to="feature")
  
  # Cleveland dotplot for ADL and ADP: Just on test set.
  ggplot(cd3) +
    geom_segment( aes(x=feature, 
                      xend=feature, 
                      y=ADL, 
                      yend=ADP), color="grey75", size=2) +
    geom_point( aes(x=feature, y=ADL),  size=3 ) + #color=rgb(0.2,0.7,0.1,0.8),
    geom_point( aes(x=feature, y=ADP),  size=3 ) + #color=rgb(0.7,0.2,0.1,0.8),
    xlab("") +
    theme_set(theme_minimal())  +
    ylab("Shapley value") +
    scale_x_discrete(labels=leg_labs) +
    geom_crossbar(data=cdCI, fatten=1, alpha=0.3, width=0.3,linetype=0,
                  aes(y=CI1, ymin=CI2, ymax=CI3, x=feature,
                      colour=type, fill=type)) +
    geom_jitter(data=cdN2,alpha=0.4,width=0.14,shape=18,
                aes(x=feature,y=value,colour=type))
}
pdf(file="DARRP_interact.pdf",width=5,height=4)
plot_compare_DARRP_N_interact(cdN_all)
dev.off()

# Library
library(fmsb)
# Create data: note in High school for Jonathan:
data <- as.data.frame(matrix( sample( 2:20 , 10 , replace=T) , ncol=10))
colnames(data) <- c("math" , "english" , "biology" , "music" , "R-coding", "data-viz" , "french" , "physic", "statistic", "sport" )
# To use the fmsb package, I have to add 2 lines to the dataframe: the max and min of each topic to show on the plot!
data <- rbind(rep(20,10) , rep(0,10) , data)
# Check your data, it has to look like this!
# head(data)
# The default radar chart 






### DARRP EXAMPLE 3 -----------------------------------------------------------
# This will be simpson's paradox (if we include it).


### Application 1 -----------------------------------------------------------
# The NHANES I dataset from [14]
Xh <- read.csv("../RL_data/X_data_with_header.csv")
y <- read.csv("../RL_data/y_data.csv", header = F)
names(y) <- "logRR"
Xh <- apply(Xh, FUN = function(x){x[is.nan(x)] <- NA; x}, MARGIN = 2)
Xh <- as_tibble(Xh)
X_dr <- remove_all_missing(Xh2, ncols = 3)
y_dr <- y[attr(X_dr, "keep"),]
dat <- cbind(y_dr, X_dr)
interesting <- c("age", "physical_activity", "systolic_blood_pressure")
fts <- which(colnames(dat) %in% interesting) - 1
fnams <- c("age", "PA", "SBP")
cdN3way <- compare_DARRRP_N_gender_3way(
  dat, sample_size = 1000, N = 100,
  features = fts, feature_names = fnams)
#save(cdN3way, file = "run1_cdN3way.dat")
plot_compare_DARRP_N(cdN1000[[1]], main = "cdN3way")






# UNUSED SNIPS ------------------------------------------------------------

# ### Radar Charts
# cd4 <- cd2 %>% 
#   filter(CI == 1 & (S %in% c("S1","S3","S5"))) %>% 
#   pivot_wider(names_from = S, values_from = value) %>%
#   select(-feature,-CI) %>% 
#   rbind(rep(max(cd2$value),d),rep(0,d),.)
# # Color vector
# colors_border=c( rgb(0.2,0.5,0.5,0.9), rgb(0.8,0.2,0.5,0.9) , rgb(0.7,0.5,0.1,0.9) )
# colors_in=c( rgb(0.2,0.5,0.5,0.4), rgb(0.8,0.2,0.5,0.4) , rgb(0.7,0.5,0.1,0.4) )
# # plot with default options:
# radarchart( cd4  , axistype=1 , 
#             #custom polygon
#             pcol=colors_border , pfcol=colors_in , plwd=4 , plty=1,
#             #custom the grid
#             cglcol="grey", cglty=1, axislabcol="grey", caxislabels=seq(0,20,5), cglwd=0.8,
#             #custom labels
#             vlcex=0.8)

#m <- dim(cdN_all)[4]
#cd <- array(dim = c(0,d))
#for (t in 1:m) {
#  cd <- cdN_all[shap_index,,,t] %>% 
#    apply(MARGIN = 1, FUN = function(x){ 
#      c(mean(x), quantile(x, probs = p)) 
#    }) %>% rbind(cd, .)
#}
#colnames(cd) <- feature_names
#cd <- cd %>% 
#  cbind(CI = 1:3, time = rep(0:(m-1), each = 3)) %>% 
#  data.frame() %>% 
#  pivot_longer(all_of(feature_names), names_to = "feature") %>% 
#  pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
#
#plt <- ggplot(data=cd, aes(x=time, y=CI1, colour=feature, fill=feature,
#                           shape=feature, linetype=feature)) + 
#  geom_point() + geom_line() +
#  geom_ribbon(aes(ymin=CI2, ymax=CI3), linetype=2, alpha=0.1) +
#  scale_x_continuous(breaks = 0:m) +
#  scale_y_continuous(name = y_name) +
#  scale_colour_discrete(labels=leg_labs) +
#  scale_shape_discrete(labels=leg_labs) +
#  scale_fill_discrete(labels=leg_labs) +
#  scale_linetype_discrete(labels=leg_labs) + 
#  theme_set(theme_minimal())
#cd <- rep(list(array(NA, dim = c(m,d))),3)
#names(cd) <- c("cd","cd_L","cd_U")
#cd$cd[t,] <- apply(vals, FUN = mean, MARGIN = 1)
#cd$cd_L[t,] <- apply(vals, FUN = quantile, MARGIN = 1, probs = p[1])
#cd$cd_U[t,] <- apply(vals, FUN = quantile, MARGIN = 1, probs = p[2])
#cd <- lapply(cd, function(x) {
#  colnames(x) <- feature_names
#  cbind(time = 1:m, x)
#  pivot_longer(data.frame(x), all_of(feature_names))
#})
#matrix(cd)
#data.frame(test, rownames(test))

#pdf("diagnostics_drift.pdf", width = 6, height = 3)
#par(mfrow = c(1,3))
