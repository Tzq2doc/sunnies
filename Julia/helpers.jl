using Statistics
using Distributions

function order_indices(x)

    sorted_indices = Dict() # dict of sorted index of each element
    println(sort(x, dims = 1))

    for (index, value) in enumerate(sort(x, dims = 1))
        sorted_indices[value] = index
    end
    
    # return the indices elements of x would have if sorted 
    return [sorted_indices[_x] for _x in x]

end

function twodim(x)
    if ndims(x) == 1
        return reshape(x, (length(x), 1))
    end
    return x 
end

function dcov(x, y)

    x = twodim(x)
    y = twodim(y)

    # Huo Szekely Appendix A
    Iˣ = order_indices(x)
    Iʸ = order_indices(y)

    sort!(x, dims=1)
    sort!(y, dims=1)

    s(l, i) = sum(l[1:i]) # Partial sum of l up until index i

    αˣ(i) = Iˣ[i] - 1
    αʸ(i) = Iʸ[i] - 1
    
    βˣ(i) = s(x, i) * (Iˣ[i] - 1)
    βʸ(i) = s(y, i) * (Iʸ[i] - 1)

    return αˣ(2), βʸ(2)


end

function dcor(x, y)

end

# TESTING 
#x = rand(Uniform(0,1), 10, 3)
y = rand(Uniform(0,1), 10, 1)
#println(cov(x, y))
#println(cov(x))
#println(cov(y))
#println(var(x))
x = [4, 8, 2]
#println(order_indices(x))
println(dcov(x,x))
println(dcov(x,y))
