using Statistics
using Distributions
using Distances


# evaluate(Euclidean(), x, y) == sqrt(sum([z^2 for z in (x-y)])) 

function distance_matrix(x, y)
    return pairwise(Euclidean(), x, y, dims=1)
end

function dcov(x, y)
    n = size(x)[1] # number of observations 
    #TODO: check that n_obs in x and y are the same?

    A = calc_mat(x, n)
    #B = calc_mat(y)
    #return 1/n/n * sum(A*B)
end

function calc_mat(x, n)
    D = distance_matrix(x, x)
    println(D)
    println(1/n * (sum(D, dims=1) - transpose(sum(D, dims=2))))

end


# TESTING 
x = rand(Uniform(0,1), 10, 3)
y = rand(Uniform(0,1), 10, 1)
#println(cov(x, y))
#println(cov(x))
#println(cov(y))
#println(var(x))
dcov(x, y)

function twodim(x)
    if ndims(x) == 1
        return reshape(x, (length(x), 1))
    end
    return x 
end


# UNIVARIATE CASE: Huo Szekely Appendix A
function order_indices(x)

    sorted_indices = Dict() # dict of sorted index of each element
    #for (index, value) in enumerate(sort(x, dims = 1))
    for (index, value) in enumerate(sort(x))
        sorted_indices[value] = index
    end
    
    # return the indices elements of x would have if sorted 
    return [sorted_indices[_x] for _x in x]

end

function dcov_univariate(x, y)

    # 1)
    Iˣ = order_indices(x)
    Iʸ = order_indices(y)

    #sort!(x, dims=1)
    #sort!(y, dims=1)
    sort!(x)
    sort!(y)

    # 2)
    s(l, i) = sum(l[1:i]) # Partial sum of l up until index i

    # 3)
    αˣ(i) = Iˣ[i] - 1
    αʸ(i) = Iʸ[i] - 1
    
    βˣ(i) = s(x, i) * (Iˣ[i] - 1)
    βʸ(i) = s(y, i) * (Iʸ[i] - 1)

    println(αˣ(2))
    println(βʸ(2))
    # 4)
    # 5)
    # 6)
    # 7)
    # 8)
    # 9)

    return 
end

function dcor(x, y)

end

#x = [4, 8, 2]
#println(order_indices(x))
#dcov_univariate(x,x)
