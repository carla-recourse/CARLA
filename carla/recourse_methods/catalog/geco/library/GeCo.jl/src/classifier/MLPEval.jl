module MLPEvaluation

using DataFrames, MLJ

export PartialMLPEval, initMLPEval, predict

const NUM_EXTRA_COL = 4
const NUM_EXTRA_FEASIBLE_SPACE_COL = 2

inplace_relu(X) = (X .= max.(X,0))
inplace_tanh(X) = (X .= tanh.(X))
inplace_logistic(X) = (X .= 1. ./ (1. .+ exp.(-X)))

const ACTIVATIONS = Dict(
    "relu" => inplace_relu,
    "tanh" => inplace_tanh,
    "logistic" => inplace_logistic,
)

struct PartialMLPEval
    orig_instance::Vector{Float64}
    coefs::Vector{Array{Float64,2}}
    intercepts::Vector{Vector{Float64}}
    activation::String
    partial_cache::Dict{BitVector,Array{Float64,2}}
end

initMLPEval(classifier, orig_instance) =
    PartialMLPEval(
        collect(Float64,orig_instance),
        classifier.coefs_, classifier.intercepts_, classifier.activation,
        Dict{BitVector,Array{Float64,1}}())

@inline function predict(clf::PartialMLPEval, df::DataFrame)
    insertcols!(df, :pred => Vector{Float64}(undef, nrow(df)), copycols=false)
    gb = groupby(df, :mod)
    g = ACTIVATIONS[clf.activation]
    for df in gb
        mod::BitVector = df[1,:mod]
        matrix::Array{Float64,2} = repeat(if haskey(clf.partial_cache, mod)
            clf.partial_cache[mod]
        else
            nmod = .!mod
            mtx = clf.orig_instance[nmod]' * clf.coefs[1][nmod,:] + clf.intercepts[1]'
            clf.partial_cache[mod] = convert(Array{Float64,2}, mtx)
        end, inner = (nrow(df),1))
        num_layer = length(clf.coefs)
        matrix += (matrixwithtype(Float64, df[!,1:end-NUM_EXTRA_COL-1][!,mod])::Array{Float64,2}) * clf.coefs[1][mod,:]
        if num_layer != 1
            g(matrix)
        else
            inplace_logistic(matrix)
        end
        for i = 2:num_layer
            matrix *= clf.coefs[i]
            for j = 1:size(matrix,1); matrix[j,:] += clf.intercepts[i]; end
            if i != num_layer
                g(matrix)
            else
                inplace_logistic(matrix)
            end
        end
        df[:,:pred] .= view(matrix,:,1) # note the order of : and 1
    end
    pred::Vector{Float64} = df[!, :pred]
    select!(df, Not(:pred))
    pred
end

# copy of MLJ.matrix to support arbitrary type conversion
function matrixwithtype(::Type{T}, table; transpose::Bool=false) where {T}
    cols = Tables.columns(table)
    n, p = Tables.rowcount(cols), ncol(table)
    if !transpose
        matrix = Matrix{T}(undef, n, p)
        for (i, col) in enumerate(Tables.Columns(cols))
            matrix[:, i] = col
        end
    else
        matrix = Matrix{T}(undef, p, n)
        for (i, col) in enumerate(Tables.Columns(cols))
            matrix[i, :] = col
        end
    end
    return matrix
end


@inline function predict(clf::PartialMLPEval, df::DataFrame, mod::BitVector)::Vector{Float64}
    g = ACTIVATIONS[clf.activation]
    matrix::Array{Float64,2} = repeat(if haskey(clf.partial_cache, mod)
        clf.partial_cache[mod]
    else
        nmod = .!mod
        mtx = clf.orig_instance[nmod]' * clf.coefs[1][nmod,:] + clf.intercepts[1]'
        clf.partial_cache[mod] = convert(Array{Float64,2}, mtx)
    end, inner = (nrow(df),1))
    # println(matrix)
    num_layer = length(clf.coefs)
    # println(MLJ.matrix(df[:,1:end]))
    # readline()
    matrix += (matrixwithtype(Float64, df[:,1:end])::Array{Float64,2}) * clf.coefs[1][mod,:]
    # println()
    # println(matrix)
    # println()
    if num_layer != 1
        g(matrix)
    else
        inplace_logistic(matrix)
    end
    # println(matrix)
    for i = 2:num_layer
        matrix *= clf.coefs[i]
        # println(matrix)
        for j = 1:size(matrix,1); matrix[j,:] += clf.intercepts[i]; end
        # println(matrix)
        if i != num_layer
            g(matrix)
        else
            inplace_logistic(matrix)
        end
        # println(matrix)
    end
    vec(matrix)
end

# https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/neural_network/_multilayer_perceptron.py#L701
function predict(coefs::Vector{Matrix{Float64}}, intercepts::Vector{Vector{Float64}}, activation::String, df::Matrix{Float64})::Vector{Float64}
    g = ACTIVATIONS[activation]
    num_layer = length(coefs)
    matrix = df
    if num_layer != 1
        g(matrix)
    else
        inplace_logistic(matrix)
    end
    for i = 1:num_layer
        matrix *= coefs[i]
        for j = 1:size(matrix,1); matrix[j,:] += intercepts[i]; end
        if i != num_layer
            g(matrix)
        else
            inplace_logistic(matrix)
        end
    end
    vec(matrix)
end

end