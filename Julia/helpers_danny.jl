using Statistics
using Distributions
using Distances
using LinearAlgebra


# evaluate(Euclidean(), x, y) == sqrt(sum([z^2 for z in (x-y)]))

function distance_matrix(x, y)
    return pairwise(Euclidean(), x, y, dims=1)
end

function calc_mat(x)
    D = distance_matrix(x, x)
    return (D .- mean(D, dims=1) .- mean(D, dims=2) .+ mean(D))
end

function dcov(x, y)
    return sqrt(dcov_sq(x, y))

end

function dcov_sq(x, y)
    A = calc_mat(x)
    B = calc_mat(y)
    #return norm(A .* B) # Wrong, don't know why
    return mean(A .* B)
end

function dcor(x, y)
    return sqrt(dcov_sq(x, y) / (dcov_sq(x, x) * dcov_sq(y, y)) )
end

# === TESTING
#x = rand(Uniform(0,1), 3, 3)
#y = rand(Uniform(0,1), 3, 3)

# --- Values from python
x = [0.828353 0.43233 0.463169; 0.397422 0.273435 0.368917; 0.483459 0.851078 0.566512]
y = [0.208873 0.815978 0.304595; 0.485556 0.256617 0.626528; 0.769359 0.552493 0.925741]
#dcov = 0.2872514947935752
#dcor = 0.9610019939406783
#x = [0.802453 0.903502 0.650715; 0.116538 0.781613 0.184347; 0.172204 0.0199412 0.687175]
#y = [0.0180497 0.259578 0.464613; 0.211949 0.278084 0.96341; 0.701463 0.740318 0.319947]
#dcov = 0.40846505175768183
#dcor = 0.9849145696875032
# ---

println(dcov(x, y))
println(dcor(x, y))

function twodim(x)
    if ndims(x) == 1
        return reshape(x, (length(x), 1))
    end
    return x
end


# UNIVARIATE CASE: Huo Szekely Appendix A
function order_indices(x)

    # Python solution without dict:
    # a = sorted(enumerate(sorted(enumerate(a), key=itemgetter(1))), key=itemgetter(1, 0))
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
