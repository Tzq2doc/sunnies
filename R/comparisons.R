source("comparison_helpers.R")

U <- DC
n <- 1e3
xgb01 <- run_evaluations(dat_concat_XOR         , U, n)
xgb09 <- run_evaluations(dat_catcat_XOR         , U, n) 
xgb08 <- run_evaluations(dat_concon_XOR         , U, n)
xgb02 <- run_evaluations(dat_unif_squared       , U, n)
xgb03 <- run_evaluations(dat_nonrandom_squared  , U, n)
xgb04 <- run_evaluations(dat_unif_squared_corr  , U, n)
xgb05 <- run_evaluations(dat_unif_independent   , U, n)
xgb06 <- run_evaluations(dat_unif_cos           , U, n) 
xgb07 <- run_evaluations(dat_unif_step          , U, n)


library(shapr)

# Produce the simulated data
dat <- dat_unif_squared(n = 1e3)
n <- nrow(dat)
train <- sample(1:n, floor(n/2))
x_train <- dat[train,-1]
colnames(x_train) <- paste0("x",1:4)
x_test <- dat[-train,-1]
colnames(x_test) <- paste0("x",1:4)
y_train <- dat[train,1]
binary <- T
if (length(unique(y_train)) > 2) {binary <- F}
obj <- if (binary) {"binary:logistic"} else {"reg:squarederror"}

# Fit XGBoost model
model <- xgboost(
  data = x_train,
  label = y_train,
  nround = 20,
  verbose = FALSE
)

# Prepare the data for explanation
explainer <- shapr(x_train, model)

# Specifying the phi_0, i.e. the expected prediction without any features
p <- mean(y_train)

# Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
# the empirical (conditional) distribution approach with bandwidth parameter sigma = 0.1 (default)
N <- 100
explanation <- explain(
  x_test[sample(1:nrow(x_test),N),],
  approach = "empirical",
  explainer = explainer,
  prediction_zero = p
)
print(explanation$dt)
expl <- explanation$dt

# Optionally set colours using RColorBrewer
library(RColorBrewer)
display.brewer.all()
cols = brewer.pal(11, "Spectral")
pal = colorRampPalette(cols)
expl$order = findInterval(expl$none, sort(expl$none))

plot(expl$x4, rep(4,nrow(expl)), 
     col=pal(nrow(expl))[expl$order], ylim = c(0,5))
points(expl$x3, rep(3,nrow(expl)),
       col=pal(nrow(expl))[expl$order])
points(expl$x2, rep(2,nrow(expl)),
       col=pal(nrow(expl))[expl$order])
points(expl$x1, rep(1,nrow(expl)),
       col=pal(nrow(expl))[expl$order])