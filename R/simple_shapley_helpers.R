# The Shapley value of a player can be broken into
# the mean of the average utility of that player
# within each team size.
simple_shapley <- function(CF, v) {
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


# We don't know the population characteristic function,
# so we use the utility function to estimate the 
# characteristic function from the data X.
estimate_characteristic_function <- function(X, utility, ...) {
  values <- list()
  players <- 1:ncol(X)
  num_players <- length(players)
  team_sizes <- 0:num_players
  
  # We now precomputes all the
  # possible values of the utility function
  for ( s in team_sizes ) {
    teams_of_size_s <- combn( players, s, simplify = F )
    for ( team in teams_of_size_s ) {
      values[[string(team)]] <- utility(X[,team], ...) 
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