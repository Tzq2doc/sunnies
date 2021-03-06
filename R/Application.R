source("datasets/simulated_datasets.R")
source("shapley_helpers.R")
source("utility_functions.R")
source("comparison_helpers.R")
source("shapley_helpers.R")
source("Applications_helpers.R")
library(xgboost)
library(dplyr)
library(naniar)
library(ggplot2)




### Python data and python model --------------------------------------------
# dat_python_tr <- read.csv("../RL_data/train_data_5050.csv") # even male/female
# dat_python_te <- read.csv("../RL_data/test_data_9010.csv") # 90% women and 10% men
# 
# sapply(dat_python_tr, class)
# xgb.DMatrix(data = dat_python_tr)
#             
# y_ptrain <- as_tibble(dat_python_tr[,ncol(dat_python_tr)])
# X_ptrain <- as_tibble(dat_python_tr[,-c(1,ncol(dat_python_tr))])
# y_ptest <- as_tibble(dat_python_te[,ncol(dat_python_te)])
# X_ptest <- as_tibble(dat_python_te[,-c(1,ncol(dat_python_te))])
# X_ptrain <- mutate_if(X_ptrain, is.factor, ~ as.integer(.x)[.x]-1L)
# X_ptest <- mutate_if(X_ptest, is.factor, ~ as.integer(.x)[.x]-1L)
# #dptr <- as.matrix(apply(dat_python_tr, FUN = as.numeric, MARGIN = 2))
# nrow(X_ptrain) # 7000
# nrow(y_ptrain) # 7000
# nrow(X_ptest)  # 4400
# nrow(y_ptest)  # 4400
# 
# # Yeah nah this looks bad
# xgb_python <- xgb.load("../RL_data/retrained_full_model.dat")
# preds <- predict(xgb_python, as.matrix(X_ptrain))
# log(preds)


# RAW data ----------------------------------------------------------------
Xh <- read.csv("../RL_data/X_data_with_header.csv")
X <- read.csv("../RL_data/X_data.csv", header = F)
y <- read.csv("../RL_data/y_data.csv", header = F)
names(y) <- "logRR"

Xh <- apply(Xh, FUN = function(x){x[is.nan(x)] <- NA; x}, MARGIN = 2)
Xh <- as_tibble(Xh)
names(Xh); nrow(Xh); nrow(X); nrow(y)
sum(Xh[["sex_isFemale"]] == F)



# C-statistic business ----------------------------------------------------
c_statistic_harrell <- function(pred, labels) {
  total <- 0 
  matches <- 0 
  for (i in 1:length(labels)) {
    for (j in 1:length(labels)) {
      if (labels[j] > 0 && abs(labels[i]) > labels[j]) {
        total <- total + 1
        if (pred[j] > pred[i]) {
          matches <- matches + 1
        }
      }
    }
  }
  return(matches/total)
}

labels <- as.numeric(y[[1]])
c_statistic_harrell(runif(nrow(y),-0.3,-0.2), labels)
sum(labels > 0)/length(labels)


# MISSINGNESS -------------------------------------------------------------
#### MISSINGNESS
# None of the y values are missing
any(!is.finite(y[[1]]))

# Visualisations
# NOTE: Proportion of patients that are missing 15+ vals: 0.5259763
make_miss_plots <- F
if (make_miss_plots) {plots <- visualise_missing(Xh); plots}

# Save the names of these 15 columns where missing values are concentrated
missv <- miss_var_summary(Xh)
high_miss <- missv[1:15,][[1]]; high_miss

# So we'll start by dropping those 15 columns
Xh2 <- select(Xh, -one_of(high_miss))

# We now plot dropped rows as a function of dropped columns
if (make_miss_plots) {plot_all_drops(Xh)}
if (make_miss_plots) {plot_all_drops(Xh2)}

# Based on the above, we could just drop 3 or 4 more columns
# (and see what they are)
X_dr <- remove_all_missing(Xh2, ncols = 3)
length(attr(X_dr, "keep"))
y_dr <- y[attr(X_dr, "keep"),]

# OR, we could drop enough columns that we don't need to drop rows
X_dc <- remove_all_missing(Xh2, p = 0)
length(attr(X_dc, "keep"))
y_dc <- y


# Splitting data proportionally with gender -------------------------------
dat <- cbind(y_dr, X_dr)
sdat <- split_dat_gender(dat, 0.5, 0.9)

# Proportion of males in training and test sets after split
sum(sdat$x_train[,"sex_isFemale"] == F)/nrow(sdat$x_train)
sum(sdat$x_test[,"sex_isFemale"] == F)/nrow(sdat$x_test)

# Proportion of data that was used in the training set
nrow(sdat$x_train)/(nrow(sdat$x_train) + nrow(sdat$x_test))


# Now for some quick analyses ---------------------------------------------
dat <- cbind(y_dr, X_dr)
interesting <- c("age", "physical_activity", "systolic_blood_pressure")
fts <- which(colnames(dat) %in% interesting) - 1
fnams <- c("age", "PA", "SBP")

ss <- 1000
NN <- 100

sdat3w <- split_dat_gender_3way(dat)

sdat <- split_dat_gender_3way(dat)
xgb <- basic_xgb_fit(sdat)
xgbt <- basic_xgb_test(xgb, sdat, valid = T)
compare_DARRP(sdat3w, xgbt, features = fts, 
              feature_names = fnams, utility = DC,
              sample_size = 100,
              valid = T)

cdN3way <- compare_DARRRP_N_gender_3way(
  dat, sample_size = 1000, N = 100,
  features = fts, feature_names = fnams)
save(cdN3way, file = "run1_cdN3way.dat")

cdN1000 <- compare_DARRRP_N_gender(
  dat, 1, 0, sample_size = ss, N = NN,
  features = fts, feature_names = fnams)
saveRDS(cdN1000, file = "run2_cdN1000.dat")
cdN0505 <- compare_DARRRP_N_gender(
  dat, 0.5, 0.5, sample_size = ss, N = NN,
  features = fts, feature_names = fnams)
saveRDS(cdN0505, file = "run2_cdN0505.dat")
cdN0901 <- compare_DARRRP_N_gender(
  dat, 0.9, 0.1, sample_size = ss, N = NN,
  features = fts, feature_names = fnams)
saveRDS(cdN0901, file = "run2_cdN0905.dat")

cdN1000 <- readRDS("run1_cdN1000.dat")
cdN0505 <- readRDS("run1_cdN0505.dat")
cdN0901 <- readRDS("run1_cdN0905.dat")

cdN1000 <- readRDS("run2_cdN1000.dat")
cdN0505 <- readRDS("run2_cdN0505.dat")
cdN0901 <- readRDS("run2_cdN0905.dat")

plot_compare_DARRP_N(cdN1000[[1]], main = "cdN1000")
plot_compare_DARRP_N(cdN0505[[1]], main = "cdN0505")
plot_compare_DARRP_N(cdN0901[[1]], main = "cdN0901")


# Other things looked at --------------------------------------------------


# Comparing some features between X_dr and X_dc
interesting <- c(
  "sex_isFemale", "age", "physical_activity", "systolic_blood_pressure")
s1 <- shapley(y_dr, X_dr[,interesting], utility = DC)
s2 <- shapley(y_dc, X_dc[,interesting], utility = DC)
s <- rbind(s1, s2)
colnames(s) <- c("sex", "age", "PA", "SBP")
barplot(s,
        xlab = "Feature", ylab = "Attribution",
        col = c("black","gray"), beside = T)

# Comparing some features between all male and all female (in X_dc)
interesting <- c(
  "sex_isFemale", "age", "physical_activity", "systolic_blood_pressure")
X_dcm <- filter(X_dc, sex_isFemale == 0)
X_dcf <- filter(X_dc, sex_isFemale == 1)
y_dcm <- y_dc[X_dc$sex_isFemale == 0,]
y_dcf <- y_dc[X_dc$sex_isFemale == 1,]
s1 <- shapley(y_dcm, X_dcm[,interesting], utility = DC)
s2 <- shapley(y_dcf, X_dcf[,interesting], utility = DC)
s <- rbind(s1, s2)
colnames(s) <- c("sex", "age", "PA", "SBP")
barplot(s,
        xlab = "Feature", ylab = "Attribution",
        col = c("black","gray"), beside = T)
legend(x = "top", legend = c("males","females"), 
       col = c("black","gray"), pch = c(15,15))




# UNUSED ------------------------------------------------------------------

#pvec <- seq(0, 1, by = 0.05)
#rows <- vector(mode = "numeric", length = length(pvec)); cols <- rows
#for (i in 1:length(pvec)) {
#  counts <- remove_all_missing(Xh, p = pvec[i], count_only = T)
#  cols[i] <- counts$cols
#  rows[i] <- counts$rows
#}
#plot(cols, rows, type = 'b', 
#     xlab = "# Dropped Columns",
#     ylab = "# Dropped Rows",
#     lab = c(15,5,7))




