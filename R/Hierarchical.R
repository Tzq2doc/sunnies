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


dat <- dat_a_few_important(n = 1e3, d1 = 4, d0 = 30)
out <- binary_tree_shap(dat)
barplot(out$s)
attr(dat, "important")


# # Quick test / checking
# targets <- list(c(1,2,3),c(4,5,6),c(7,8,9))
# dat <- dat_a_few_important(n = 1e3, d1 = 4, d0 = 30)
# barplot(shapley2(dat[,c(1,2,3,4,6,7)], utility = DC))
# barplot(shapley(dat[,c(1,2,3,4,6,7)], utility = DC))
# barplot(shapley2(dat, utility = DC,
#                  targets = targets))
# dat1 <- dat_unif_squared()
# X <- dat1[,-1] #dat
# y <- dat1[,1]
# barplot(shapley2(y,X, utility = DC, targets = NULL))
# barplot(shapley(y,X, utility = DC))

# Our final goal will be to identify the k most important features

# My first tree chooser is just simple. It does a kind of binary search.
# It just keeps the branch with the highest dependence. Later, I only
# want to drop when the difference is high.
binary_tree_step <- function(indices) {
  ni <- length(indices)
  if (ni %% 2 == 0) {
    return(list(indices[1:(ni/2)], indices[(ni/2+1):ni]))  
  } else {
    s <- (ni-1)/2
    return(list(c(indices[1:s],ni), indices[(s+1):ni]))
  }
}

binary_tree_shap <- function(dat, k = 4) {
  d <- ncol(dat)-1; i <- 0
  targets <- binary_tree_step(1:d)
  bshaps <- matrix(0, nrow = 2*log2(d), ncol = 2)
  directions <- vector(mode = "integer", length = 2*log2(d))
  target_sizes <- vector(mode = "integer", length = 2*log2(d)) 
  while (length(targets[[1]]) > k) {
    i <- i + 1
    target_sizes[i] <- length(targets[[1]])
    bshaps[i,] <- shapley2(dat, utility = DC, targets = targets)
    directions[i] <- 1 + (bshaps[i,2] > bshaps[i,1])
    targets <- binary_tree_step(targets[[directions[i]]])
    target_sizes[i] <- length(targets[[1]])
  }
  bshaps <- bshaps[1:i,]
  directions <- directions[1:i]
  target_sizes <- target_sizes[1:i]
  #bshaps; directions; target_sizes
  final_targets <- as.list(unlist(targets))
  s <- shapley2(dat, utility = DC, targets = final_targets)
  names(s) <- unlist(final_targets)
  return(list(s=s, bshaps = bshaps, directions = directions,
              target_sizes = target_sizes))
}




# Investigating dropping odds ---------------------------------------------

binary_tree_counter <- function(d) {
  i <- 1
  rvec <- vector(mode = "numeric", length = ceiling(2*log2(d)))
  dvec <- vector(mode = "numeric", length = ceiling(2*log2(d)))
  while (d > 3) {
    if (d %% 2 == 0) { 
      dvec[i] <- d/2 
    } else { 
      dvec[i] <- (d-1)/2
      rvec[i] <- d
    }
    d <- dvec[i]
    i <- i + 1
  }
  list(nr = sum(rvec != 0), ns = sum(dvec != 0))
}

# Cool fractal thing for the drop bits
Nd <- 10000
rem <- vector(mode = "numeric", length = Nd)
spl <- vector(mode = "numeric", length = Nd)
for (d in 1:Nd) {
  out <- binary_tree_counter(d)
  rem[d] <- out$nr
  spl[d] <- out$ns
}
plot(1:Nd,rem, type = 'l')
plot(1:Nd,spl, type = 'l')

# utility <- DC
# 
# 
# if (missing(targets)) {targets <- as.list(1:ncol(X))}
# values <- list()
# players <- 1:length(targets)
# num_players <- length(players)
# team_sizes <- 0:num_players
# 
# for ( s in team_sizes ) {
#   teams_of_size_s <- combn( players, s, simplify = F )
#   for ( team in teams_of_size_s ) {
#     team <- combine(targets[team])
#     Xs <- X[,team,drop = F]
#     values[[access_string(team)]] <- utility(Xs, y = y) 
#   }
# }
# 
# CF <- function(t){values[[access_string(t)]]}
# attr(CF, "players") <- players
# attr(CF, "targets") <- targets
# 
# v <- 1
# players <- environment(CF)$players[-v]
# targets <- environment(CF)$targets
# num_players <- length(players)
# team_sizes <- 0:num_players
# value <- 0
# for ( s in team_sizes ) {
#   value_s <- 0
#   teams_of_size_s <- if (length(players) != 1) {
#     combn(players, s, simplify = F)} else 
#     {list(players)}
#   for ( team in teams_of_size_s ) {
#     Sv <- combine(targets[c(v,team)])
#     S <- combine(targets[team])
#     value_in_team <- CF(Sv) - CF(S)
#     value_s <- value_s + value_in_team
#   }
#   average_value_s <- value_s/length(teams_of_size_s)
#   value <- value + average_value_s
# }
# average_value <- value/length(team_sizes)
# average_value 
# 
# # We'll first need to rewrite the shapley function to handle sets of features



# HELPERS -----------------------------------------------------------------

# NOTE: targets can be a named list

estimate_CF2 <- function(X, utility, drop_method = "actual", 
                         targets = NULL, ...) {
  values <- list()
  if (is.null(targets)) {targets <- as.list(1:ncol(X))}
  players <- 1:length(targets)
  num_players <- length(players)
  team_sizes <- 0:num_players
  
  # We now precompute all the
  # possible values of the utility function
  if ( tolower(drop_method) == "actual" ) {
    for ( s in team_sizes ) {
      teams_of_size_s <- combn( players, s, simplify = F )
      for ( team in teams_of_size_s ) {
        team <- combine(targets[team])
        Xs <- X[,team,drop = F]
        values[[access_string(team)]] <- utility(Xs, ...) 
      }
    }
  } else if ( tolower(drop_method) == "mean" ) {
    for ( s in team_sizes ) {
      teams_of_size_s <- combn( players, s, simplify = F )
      for ( team in teams_of_size_s ) {
        team <- combine(targets[team])
        Xs <- mean_drop(X, team)
        values[[access_string(team)]] <- utility(Xs, ...) 
      }
    }
  }
  # We created some bindings in this environment 
  # and we are now returning a function that 
  # permantently has access to this environment,
  # so we can access this environment from anywhere
  CF <- function(t){values[[access_string(t)]]}
  attr(CF, "players") <- players
  attr(CF, "targets") <- targets
  return(CF)
}


shapley2 <- function(y, X, utility, v, CF, drop_method = "actual", 
                     targets = NULL, ...) {
  if ( !is.matrix(y) ) {y <- as.matrix(y)}
  if (any(!is.finite(y))) {stop(
    paste0("shapley can only handle finite numbers at this time, ",
           "please check y for NA, NaN or Inf"))}
  if (!missing(CF)) {
    if (missing(v)) {
      v <- attr(CF,"players")
    }
    return(shapley_vec(CF, v))
  }
  if ( ncol(y) > 1 & missing(X) ) {
    X <- y[,-1, drop = F]
    y <- y[, 1, drop = F]
  }
  if ( !is.matrix(X) ) {X <- as.matrix(X)}
  if ( any(!is.finite(X)) ) {stop(
    paste0("shapley can only handle finite numbers at this time, ",
           "please check X for NA, NaN or Inf"))}
  if (missing(v)) { if (is.null(targets)) {v <- 1:ncol(X)} else {
    v <- 1:length(targets)  
  }}
  
  CF <- estimate_CF2(X, utility, drop_method = drop_method, 
                    targets = targets, y = y,...)
  sv <- shapley_vec2(CF, v)
  if (is.null(targets)) {names(sv) <- colnames(X)} else {
    names(sv) <- names(targets)
  }
  return(sv)
}


shapley_v2 <- function(CF, v) {
  players <- environment(CF)$players[-v]
  targets <- environment(CF)$targets
  num_players <- length(players)
  team_sizes <- 0:num_players
  value <- 0
  for ( s in team_sizes ) {
    value_s <- 0
    teams_of_size_s <- if (length(players) != 1) {
      combn(players, s, simplify = F)} else 
      {list(players)}
    for ( team in teams_of_size_s ) {
      Sv <- combine(targets[c(v,team)])
      S <- combine(targets[team])
      value_in_team <- CF(Sv) - CF(S)
      value_s <- value_s + value_in_team
    }
    average_value_s <- value_s/length(teams_of_size_s)
    value <- value + average_value_s
  }
  average_value <- value/length(team_sizes)
  return(average_value)
}


shapley_vec2 <- Vectorize(shapley_v2, "v")
