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


targets <- list(c(1,2,3),c(4,5,6),c(7,8,9))
dat <- dat_a_few_important(n = 1e3, d1 = 4, d0 = 10)
barplot(shapley2(dat[,c(1,2,3,4,6,7)], utility = DC))
barplot(shapley(dat[,c(1,2,3,4,6,7)], utility = DC))

barplot(shapley2(dat, utility = DC,
                 targets = targets))


# Our final goal will be to identify the k most important features
dat1 <- dat_unif_squared()
X <- dat1[,-1] #dat
y <- dat1[,1]
barplot(shapley2(y,X, utility = DC, targets = NULL))
barplot(shapley(y,X, utility = DC))

targets <- list(c(1,2,3),c(4,5,6),c(7,8,9))


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
