using Distributions
using Combinatorics
using LinearAlgebra
using Statistics
using Random

#using PkgTemplates
#generate("MyPackage", Template(;user="My Name", dir=".", dev=false))
#Module Shapley

# We need to replace R² with the distance correlation
function CF(Z, s, cf_name)

    if cf_name=="R²"
        Cₙ_ = cor(Z)
        Cₙ(u) = [Cₙ_[i,j] for i in u .+ 1, j in u .+ 1]
        # Equation (4) (taking care of empty s here)
        R²(s) = (length(s) > 0) ? 1 - det(Cₙ(vcat(0,s)))/det(Cₙ(s)) : 0
        CF_value = R²(s)
    
    elseif cf_name=="dcor"
        CF_value = 1

    else
        throw(DomainError(cf_name, "not implemented"))
    end

  return CF_value 
end

function shapley(Z; vals = 1:size(Z)[2]-1, cf_name="R²")
  d, n = size(Z)[2]-1, size(Z)[1]

  # Equation (9) (pre-compute ω_ for efficiency)
  ω_ = [factorial(i)/factorial(d, d-i-1) for i in 0:(d-1)]
  ω(s) = ω_[length(s) + 1]
  S(j) = deleteat!(collect(1:d), j)

  #V(j) = sum([ω(s)*(R²(vcat(j,s)) - R²(s)) for s in powerset(S(j))])
  V(j) = sum([ω(s)*(CF(Z, vcat(j,s), cf_name) - CF(Z, s, cf_name)) for s in powerset(S(j))])

  # Calculate all the shapley values using Equation (9)
  return map( x -> V(x), vals )
end


#function 𝕮(Z)
#end

#end

### TESTING
d = 3
c = 0.2
n = 500
cf = "R²"
#cf = "dcor"
M0(c,d) = [Float64(c) + (i == j)*(1-Float64(c)) for i in 1:(d+1), j in 1:(d+1)]
mvn(M) = MvNormal(M)
Z = rand(mvn(M0(c,d)), n)'
shap = shapley(Z, cf_name=cf)
println(shap)








# #### TESTING
# # Parameters
# Random.seed!(0)
# d, n = 2, 1000
# X = randn( n, d )
# y = X * collect(0:2:(2d-2)) + randn( n )
# Z = hcat(y,X)
# @time s, v = calc_shapley(Z)
# @time s, v = calc_shapley(Z, with_cov = true)
#
# # Check that compiler figures out all the types
# @code_warntype calc_shapley(Z)
# # Benchmarking
# using BenchmarkTools
# @benchmark calc_shapley(Z)
