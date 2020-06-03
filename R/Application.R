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


# Python data and python model --------------------------------------------

dat_python_tr <- read.csv("../RL_data/train_data_5050.csv") # even male/female
dat_python_te <- read.csv("../RL_data/test_data_9010.csv") # 90% women and 10% men

y_ptrain <- as_tibble(dat_python_tr[,ncol(dat_python_tr)])
X_ptrain <- as_tibble(dat_python_tr[,-ncol(dat_python_tr)])
y_ptest <- as_tibble(dat_python_te[,ncol(dat_python_te)])
X_ptest <- as_tibble(dat_python_te[,-ncol(dat_python_te)])
X_ptrain <- mutate_if(X_ptrain, is.factor, ~ as.integer(.x)[.x]-1L)
X_ptest <- mutate_if(X_ptest, is.factor, ~ as.integer(.x)[.x]-1L)
nrow(X_ptrain) # 7000
nrow(y_ptrain) # 7000
nrow(X_ptest)  # 4400
nrow(y_ptest)  # 4400

# Yeah nah this looks bad
xgb_python <- xgb.load("../RL_data/full_model.dat")
predict(xgb_python, as.matrix(X_ptrain))



# RAW data ----------------------------------------------------------------



Xh <- read.csv("../RL_data/X_data_with_header.csv")
X <- read.csv("../RL_data/X_data.csv", header = F)
y <- read.csv("../RL_data/y_data.csv", header = F)
names(y) <- "logRR"

Xh <- apply(Xh, FUN = function(x){x[is.nan(x)] <- NA; x}, MARGIN = 2)
Xh <- as_tibble(Xh)
names(Xh)
nrow(Xh)
nrow(X)
nrow(y)
sum(Xh[["sex_isFemale"]] == F)






#### MISSINGNESS
# None of the y values are missing
any(!is.finite(y[[1]]))

# Visualisations
# NOTE: Proportion of patients that are missing 15+ vals: 0.5259763
plots <- visualise_missing(Xh); plots

# Save the names of these 15 columns where missing values are concentrated
missv <- miss_var_summary(Xh)
high_miss <- missv[1:15,][[1]]; high_miss

# So we'll start by dropping those 15 columns
Xh2 <- select(Xh, -one_of(high_miss))

# We now plot dropped rows as a function of dropped columns
plot_all_drops(Xh)
plot_all_drops(Xh2)

# Based on the above, we could just drop 3 or 4 more columns
# (and see what they are)
X_dr <- remove_all_missing(Xh2, ncols = 3)
length(attr(X_dr, "keep"))
y_dr <- y[attr(X_dr, "keep"),]

# Or, we could drop enough columns that we don't need to drop rows
X_dc <- remove_all_missing(Xh2, p = 0)
length(attr(X_dc, "keep"))
y_dc <- y

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

# Choose X and y here
dat <- as.matrix(cbind(y_dr,X_dr))
sdat <- split_dat(dat)

xgb <- basic_xgb_fit(sdat); save(xgb)
xgbt <- basic_xgb_test(xgb, sdat)
fts <- which(names(X) %in% interesting)
diagn <- diagnostics(sdat, xgbt, plot = "all",
                     features = fts,
                     feature_names = c("sex", "age", "PA", "SBP"))

s3 <- shapley(sdat$y_train, sdat$x_train[,1:2], utility = DC)
s3


split_dat_gender <- function(dat, p1f, p2f) {
  n <- nrow(dat)
  male_index <- (dat[["sex_isFemale"]] == F)
  nm <- sum(male_index)
  n1m <- calc_n1m(n, nm, p1f, p2f)
  X_m <- dat[male_index,-1]
  y_m <- dat[male_index,1]
  X_f <- dat[!male_index,-1]
  y_f <- dat[!male_index,1]
  
  
}


# Calculates number of males n1m in the training set, where it is
# assumed that we do not want to discard anybody.
# n := number of people overall
# nm := number of males overall
# p1f := proportion of females to males in the training set
# p2f := proportion of females to males in the test set
calc_n1m <- function(n, nm, p1f, p2f) {
  K1 <- p1f/(1-p1f)
  K2 <- p2f/(1-p2f)
  (n - nm*(K2+1))/(K1-K2)
}

n <- 14264
nm <- 5765
p1f <- 0.5
p2f <- 0.9
calc_n1m(n, nm, p1f, p2f)






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



# # If we want to increase size of training set we need to drop females.
# # Calculate number of discarded females df as a function of
# # number of people in the training set, n1
# # where the total number of people n and males nm are fixed:
# n1_calc_df <- function(n1, p1f, p2f, n, nm) {
#   K1 <- p1f/(1-p1f)
#   K2 <- p2f/(1-p2f)
#   df <- n - n1*(K1-K2)/(K1+1) - nm*(K2+1)
#   return(df*(df > 0))
# }
# n1 <- 10000:14000
# plot(n1, calc_df(n1, 0.5, 0.9, n, 5765), type = 'l')
