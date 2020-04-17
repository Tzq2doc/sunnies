
### shapleyN calls the shapley function for many
# iterations of simulated data
## Parameter names
# n: sample size
# d: number of features
# N: number of iterations simulating data
shapleyN <- function(utility, N, n, d) {
  results <- matrix(0, nrow = N, ncol = d)
  for ( i in 1:N ) {
    X <- matrix(runif(n*d,-1,1), n, d)
    y <- X^2 %*% (2*(0:(d-1)))
    CF_i <- estimate_characteristic_function(X, utility, y = y)
    results[i,] <- shapley_(CF_i, 1:d)
  }
  results
}


# The Shapley value of a player can be broken into
# the mean of the average utility of that player
# within each team size.
shapley <- function(CF, v) {
  players <- environment(CF)$players[-v]
  num_players <- length(players)
  team_sizes <- 0:num_players
  value <- 0
  for ( s in team_sizes ) {
    value_s <- 0
    teams_of_size_s <- combn(players, s, simplify = F)
    for ( team in teams_of_size_s ) {
      value_in_team <- CF(c(v,team)) - CF(team)
      value_s <- value_s + value_in_team
    }
    average_value_s <- value_s/length(teams_of_size_s)
    value <- value + average_value_s
  }
  average_value <- value/length(team_sizes)
  return(average_value)
}

shapley_ <- Vectorize(shapley, "v")


# We don't know the population characteristic function,
# so we use the utility function to estimate the 
# characteristic function from the data X.
estimate_characteristic_function <- function(X, utility, ...) {
  values <- list()
  players <- 1:ncol(X)
  num_players <- length(players)
  team_sizes <- 0:num_players
  
  # We now precompute all the
  # possible values of the utility function
  for ( s in team_sizes ) {
    teams_of_size_s <- combn( players, s, simplify = F )
    for ( team in teams_of_size_s ) {
      values[[string(team)]] <- utility(X[,team,drop = F], ...) 
    }
  }
  # We created some bindings in this environment 
  # and we are now returning a function that 
  # permantently has access to this environment,
  # so we can access this environment from anywhere
  return(function(t){values[[string(t)]]})
}

# This function converts teams into strings so we can look
# them up in the characteristic function, a bit like a dictionary.
string <- function(team) {paste0("-", sort(team), collapse = "")}