using Statistics
using Distributions
using Distances
using LinearAlgebra
using InteractiveUtils
using BenchmarkTools

function distance_matrix(x, y)
    return pairwise(Euclidean(), x, y, dims=1)
end

#function calc_mat(x)::Array{<:Number,2}
function calc_mat(x)::Array{Float64,2} # Faster but more memory expensive?
    D = distance_matrix(x, x) # size(x)[1]xsize(x)[1] Array{Float64,2} - 
    return (D .- mean(D, dims=1) .- mean(D, dims=2) .+ mean(D))
end

function dcov(x, y)::Float64
#function dcov(x, y)<:Number # Doesnt work. why?
    
    # Slower:
    #A::Array{<:Number,2} = calc_mat(x)
    #B::Array{<:Number,2} = calc_mat(y)
    
    A = calc_mat(x) # size(x)[1]xsize(x)[1] Array{Float64,2}
    B = calc_mat(y)# size(y)[1]xsize(y)[1] Array{Float64,2}
    return sqrt(mean(A .* B))
end

function dcor(x, y)::Float64
#function dcor(x, y)<:Number # Doesnt work. why?
    return (dcov(x, y)) / (sqrt(dcov(x, x) * dcov(y, y)))
end

# === TESTING 
#x = rand(Uniform(0,1), 3, 3)
#y = rand(Uniform(0,1), 3, 3)

x = rand(Uniform(0,1), 10, 5)
y = rand(Uniform(0,1), 10, 1)

# --- Values from python
#x = [0.828353 0.43233 0.463169; 0.397422 0.273435 0.368917; 0.483459 0.851078 0.566512]
#y = [0.208873 0.815978 0.304595; 0.485556 0.256617 0.626528; 0.769359 0.552493 0.925741]
#dcov = 0.2872514947935752
#dcor = 0.9610019939406783
#x = [0.802453 0.903502 0.650715; 0.116538 0.781613 0.184347; 0.172204 0.0199412 0.687175]
#y = [0.0180497 0.259578 0.464613; 0.211949 0.278084 0.96341; 0.701463 0.740318 0.319947]
#dcov = 0.40846505175768183
#dcor = 0.9849145696875032
# ---

@code_warntype dcor(x, y)
