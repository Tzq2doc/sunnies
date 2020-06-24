using Distributions
using Combinatorics
using LinearAlgebra
using Statistics
using Random

include("helpers.jl")

function CF(Z, s, cf_name)
    x = Z[:, s] #?
    y = Z[:, size(Z)[2]] #?

    if cf_name=="R²"
        # Danny, how do you tell R² what's the label?
        Cₙ_ = cor(Z)
        Cₙ(u) = [Cₙ_[i,j] for i in u .+ 1, j in u .+ 1]
        # Equation (4) (taking care of empty s here)
        R²(s) = (length(s) > 0) ? 1 - det(Cₙ(vcat(0,s)))/det(Cₙ(s)) : 0 # if s is empty, return 0
        CF_value = R²(s)
    
    elseif cf_name=="dcor"
        CF_value = dcor(x, y)

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
    
    V(j) = sum([ω(s)*(CF(Z, vcat(j,s), cf_name) - CF(Z, s, cf_name)) for s in powerset(S(j))])
    
    # Calculate all the shapley values using Equation (9)
    #return map( x -> V(x), vals )
    return map(V, vals)
end


# === TESTING
function test_shapley()
    x =   [0.755635    0.345446  -0.688384
          -1.11828    -1.68771   -0.596009
          -2.38458    -0.121187  -0.815811
           0.0603389   0.656752  -0.327731
           0.826708    0.97178   -0.164207
          -1.45166    -0.579462   1.4277  
          -0.720464    0.60213   -0.244417
           0.232181    0.96931    0.736506
          -1.3447      1.12004   -1.21429 
          -0.405403   -0.562174   0.581327]
    y = [  0.23258329322511723
           0.2833107386114472 
          -1.2564352832581345 
          -0.5874108883625386 
          -0.9447399412637602 
          -0.23829487490839948
          -1.5497635035663024 
           0.7354636701191419 
           2.0315107752106654 
           1.6625315369928169]
    z = hcat(x, y)
    # Compare to values from python
    cf = "dcor"
    dcor_sunnies_python = [0.13270700386803647, 0.16079441322043492, 0.21536146625667366]
    @assert (map(x -> round(x, digits=6), shapley(z, cf_name=cf)) == map(x -> round(x, digits=6), dcor_sunnies_python))

    cf = "R²"
    println(shapley(z, cf_name=cf))
    #r2_sunnies_python = [0.0028289722726558275, 0.002378728006757924, 0.0001955632692127729]
    #@assert (map(x -> round(x, digits=6), shapley(z, cf_name=cf)) == map(x -> round(x, digits=6), r2_sunnies_python))

end
test_shapley()
#d = 3
#c = 0.2
#n = 10#500
#cf = "R²"
#cf = "dcor"
#M0(c,d) = [Float64(c) + (i == j)*(1-Float64(c)) for i in 1:(d+1), j in 1:(d+1)]
#mvn(M) = MvNormal(M)
##
## what's x, what's y?
#Z = rand(mvn(M0(c,d)), n)'
#
#shap = shapley(Z, cf_name=cf)
#println(shap)
#
#cf = "R²"
#println(shapley(Z, cf_name=cf))








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
