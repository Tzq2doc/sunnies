using Statistics
using Distributions

function order_indices(x)

    x_ = hcat(a, 1:length(a)) # line up elements of x next to their indices 
    x_ = x_[sortperm(x_[:, 1]), :] # sort indices by the permutation which would sort x
    return x_[: ,2] # return the indices elements of x would have if sorted 

end
#Huo Szekely Appendix A
function dcov(x, y)
    Iˣ = sortperm(x) # indices before sorting
    Iʸ = sortperm(y) # indices before sorting

    # Sort x and y
    sort!(x, dims=1)
    sort!(y, dims=1)

    s(l, i) = sum(l[1:i]) # Partial sum of l up until index i
    alphaˣᵢ = Iˣ[i] - 1

end

function dcor(x, y)

end

# TESTING 
x = rand(Uniform(0,1), 10, 3)
y = rand(Uniform(0,1), 10, 1)
#println(cov(x, y))
println(cov(x))
#println(cov(y))
println(var(x))
