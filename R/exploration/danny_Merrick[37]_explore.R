source("utility_functions.R")
source("shapley_helpers.R")
source("datasets/simulated_datasets.R")

n <- 1e2
p <- c(0.1,0,0.4,0.5)

# Generate feature observations
outcome_table <- matrix(nrow = 4, byrow = T,
                        c(0,0,
                          0,1,
                          1,0,
                          1,1))
i_sample <- sample(1:4, n, replace = T, prob = p)
dat <- outcome_table[i_sample,]
colnames(dat) <- c("male","lift")

# Generate response observations
fmale <- dat[,"male"]
fboth <- as.integer(dat[,"male"] & dat[,"lift"])

### Add corruption
# prob_corrupt <- 0.05 # Note about 1/4 of corruptions will be identity
# num_corruptions <- as.integer(prob_corrupt*n)
# i_corrupt <- sample(1:n, num_corruptions)
# dat[i_corrupt,] <- outcome_table[i_corrupt %% 4 + 1, ]

### Interesting stuff
DC(fboth, dat[,"lift"])
DC(fboth, dat[,"male"])
DC(fmale, dat[,"lift"]) 
DC(fmale, dat[,"male"])
cor(fboth, dat[,"lift"])
cor(fboth, dat[,"male"])
cor(fmale, dat[,"lift"])
cor(fmale, dat[,"male"])


sdc1 <- shapley(cbind(fboth, dat), utility = DC)
sdc2 <- shapley(cbind(fmale, dat), utility = DC)
sdc1 # fboth: All value incorrectly attributed to lift
sdc2 # fmale: All value correctly attributed to male

### The differences between the shapley values equal
### the differences in the pairwise correlations
sdc1; sdc1[2] - sdc1[1]
DC(fboth, dat[,"lift"]) - DC(fboth, dat[,"male"])
sdc2; sdc2[2] - sdc2[1]
DC(fmale, dat[,"lift"]) - DC(fmale, dat[,"male"])
