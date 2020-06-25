using Statistics
using Distributions
using Distances
using LinearAlgebra


# evaluate(Euclidean(), x, y) == sqrt(sum([z^2 for z in (x-y)])) 

function distance_matrix(x)
    if isa(x, Array{Float64,1})
        x = reshape(x, size(x)[1], 1)
    end
    return pairwise(Euclidean(), x, x, dims=1)
end

function calc_mat(x)
    D = distance_matrix(x)
    return (D .- mean(D, dims=1) .- mean(D, dims=2) .+ mean(D))
end

function dcov(x, y)
    A = calc_mat(x)
    B = calc_mat(y)
    return sqrt(mean(A .* B))
end

function dcor(x, y)
    if length(x) > 0 && length(y) > 0
        return (dcov(x, y)) / (sqrt(dcov(x, x) * dcov(y, y)))
    end
    return 0
end

function aidc(x, y)
    if length(x) == 0
        return 0
    end
    cov_y = cov(y)
    cov_x = cov(x) # NB! This is cov(x.T) in python

    #TODO:What if x is empty? then, cov(x)=-0.0 in Julia
    if cov_x == -0.0
        x_trans = x
    elseif size(cov_x) == ()
        inv_cov_x = 1.0/cov_x
        x_trans = x * sqrt(inv(cov_x))
    else
        sqrt_inv_cov_x = sqrt(inv(cov_x))
        if sqrt_inv_cov_x isa Array{Complex{Float64},2}
            x_trans = x * convert(Array{Float64,2}, sqrt_inv_cov_x)
        else
            x_trans = x * sqrt_inv_cov_x
        end
    end

    inv_cov_y = 1 ./ cov_y
    y_trans = y * sqrt(inv_cov_y)

    return dcor(y_trans, x_trans)
end

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

function test_order_indices()
    x = [4, 8, 2]
    @assert order_indices(x) == [2, 3, 1]
end

# === Test functions
function test_dcov()
    # Values from python
    x = [0.828353 0.43233 0.463169; 0.397422 0.273435 0.368917; 0.483459 0.851078 0.566512]
    y = [0.208873 0.815978 0.304595; 0.485556 0.256617 0.626528; 0.769359 0.552493 0.925741]
    dcov_true = 0.2872514947935752

    @assert round(dcov_true, digits=8) == round(dcov(x, y), digits=8)

    x = [0.802453 0.903502 0.650715; 0.116538 0.781613 0.184347; 0.172204 0.0199412 0.687175]
    y = [0.0180497 0.259578 0.464613; 0.211949 0.278084 0.96341; 0.701463 0.740318 0.319947]
    dcov_true = 0.40846505175768183
    
    @assert round(dcov_true, digits=8) == round(dcov(x, y), digits=8)

    println("DCOV function ok!")
end

function test_dcor()
    # Values from python
    x = [0.828353 0.43233 0.463169; 0.397422 0.273435 0.368917; 0.483459 0.851078 0.566512]
    y = [0.208873 0.815978 0.304595; 0.485556 0.256617 0.626528; 0.769359 0.552493 0.925741]
    dcor_true = 0.9610019939406783

    @assert round(dcor_true, digits=8) == round(dcor(x, y), digits=8)

    x = [0.802453 0.903502 0.650715; 0.116538 0.781613 0.184347; 0.172204 0.0199412 0.687175]
    y = [0.0180497 0.259578 0.464613; 0.211949 0.278084 0.96341; 0.701463 0.740318 0.319947]
    dcor_true = 0.9849145696875032
    
    @assert round(dcor_true, digits=8) == round(dcor(x, y), digits=8)

    println("DCOR function ok!")
end

function test_aidc()

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
    x1 = [ -0.341371   -0.133888  -1.26302   -0.467231    0.644707 
          -0.0524715  -0.829657   0.600282   0.78509     1.49509  
           0.081209   -1.60534   -0.467177  -0.715523   -1.31282  
           0.597182    1.67758   -0.149649   0.198237    3.2101   
          -0.782839    1.24472   -0.537776  -0.774976    0.611209 
          -0.942106    1.4796     0.327017   1.66083    -0.0437138
           0.674375   -1.82007    0.314038  -0.0269082   0.13076  
           0.120167   -0.495699   2.79102   -0.0135147  -0.80674  
          -1.3133     -0.678867  -0.125036   0.450812   -0.550585 
          -0.358026    0.763886  -0.642018  -1.53592     0.910746 
         ]
    y1 = [  
         0.31699426810558795 
         1.34181444337193    
        -0.508878033643789   
         2.6854023615295426  
        -0.7195432425919883  
         0.4681875927260907  
         0.027206421899317337
         0.6396687066390925  
         0.203168750153398   
        -1.0873217125552856]
    aidc_python = 0.5486719473485777
    aidc_python1 = 0.7678647928797977
    @assert round(aidc(x, y), digits=8) == round(aidc_python, digits=8)
    @assert round(aidc(x1, y1), digits=8) == round(aidc_python1, digits=8)
    println("AIDC function ok!")
end
#test_aidc()

# === TESTING 
#dcov_univariate(x,x)

#x = rand(Uniform(0,1), 3, 3)
#y = rand(Uniform(0,1), 3, 3)
    
