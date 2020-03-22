players <- 1:4
players <- players[-2]
num_players <- length(players)
team_sizes <- 0:num_players
for ( s in team_sizes ) {
  teams_of_size_s <- combn(players, s, simplify = F)
  for ( team in teams_of_size_s ) {
    print(team)
  }
}

