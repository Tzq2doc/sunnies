source("comparison_helpers.R")

U <- DC
n <- 1e3
xgb01 <- run_evaluations(dat_continuous_XOR     , U, n)
xgb02 <- run_evaluations(dat_unif_squared       , U, n)
xgb03 <- run_evaluations(dat_nonrandom_squared  , U, n)
xgb04 <- run_evaluations(dat_unif_squared_corr  , U, n)
xgb05 <- run_evaluations(dat_unif_independent   , U, n)
xgb06 <- run_evaluations(dat_unif_cos           , U, n) 
xgb07 <- run_evaluations(dat_unif_step          , U, n)
xgb08 <- run_evaluations(dat_unif_XORlike       , U, n) 
xgb09 <- run_evaluations(dat_categorical_XOR    , U, n) 
xgb10 <- run_evaluations(dat_continuous_XOR     , U, n)