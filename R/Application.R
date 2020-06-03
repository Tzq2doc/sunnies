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

# Choose X and y here
dat <- as.matrix(cbind(y_dr,X_dr))
sdat <- split_dat(dat)

xgb <- basic_xgb_fit(sdat)
xgbt <- basic_xgb_test(xgb, sdat)
fts <- which(names(X) %in% interesting)
diagn <- diagnostics(sdat, xgbt, plot = "all",
                     features = fts,
                     feature_names = c("sex", "age", "PA", "SBP"))

s3 <- shapley(sdat$y_train, sdat$x_train[,1:2], utility = DC)
s3

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
