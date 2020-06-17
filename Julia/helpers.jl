using Statistics
using Distributions

function order_indices(x)
    sorted_indices = Dict() # dict of sorted index of each element
    for (index, value) in enumerate(sort(x))
        sorted_indices[value] = index
    end
    
    # return the indices elements of x would have if sorted 
    return [sorted_indices[_x] for _x in x]

end

function dcov(x, y)
    # Huo Szekely Appendix A
    #
    Iˣ = order_indices(x) # indices if sorted
    Iʸ = order_indices(y) # indices if sorted 

    # Sort x and y
    sort!(x, dims=1)
    sort!(y, dims=1)

    s(l, i) = sum(l[1:i]) # Partial sum of l up until index i
    alphaˣᵢ = Iˣ[i] - 1

end

function dcor(x, y)

end

# TESTING 
#x = rand(Uniform(0,1), 10, 3)
#y = rand(Uniform(0,1), 10, 1)
#println(cov(x, y))
#println(cov(x))
#println(cov(y))
#println(var(x))
