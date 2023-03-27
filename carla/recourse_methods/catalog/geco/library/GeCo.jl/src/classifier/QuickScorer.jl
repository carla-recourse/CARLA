module QuickScorer

using MLJModels, MLJ
using DecisionTree
using DataFrames
using SIMD

@generated function Base.:&(a::NTuple{N,UInt64}, b::NTuple{N,UInt64})::NTuple{N,UInt64} where {N}
    quote
        tuple($(ntuple(i -> :(a[$i] & b[$i]), N)...))
    end
end

@generated function Base.trailing_zeros(x::NTuple{N,UInt64}) where {N}
    expr = 64 * N
    for i = N:-1:1
        expr = quote
            if x[$i] != 0
                $(64 * (i-1)) + trailing_zeros(x[$i])
            else
                $expr
            end
        end
    end
    expr
end

@generated function allones(::Val{N}) where {N}
    quote
        tuple($(ntuple(i -> ~UInt(0), N)...))
    end
end

# WSIZE is the number of Int64's needed to represent the leaves of a tree.
struct QSEnsemble{S,T,WSIZE}
    feat_count::Int
    tree_count::Int
    leaves::Vector{T}
    leave_offset::Vector{Int}
    feat_offset::Vector{Int}
    featvals::Vector{S}
    treeids::Vector{Int}
    bvs::Vector{NTuple{WSIZE, UInt64}}
    vbvs::Vector{NTuple{WSIZE, UInt64}}
    initv::Vector{NTuple{WSIZE, UInt64}}
    # we put v here so we need less frequent gc
    v::Vector{NTuple{WSIZE, UInt64}}
end

calcleafcnt(tree::DecisionTree.Node{S,T}) where {S,T} =
    calcleafcnt(tree.left) + calcleafcnt(tree.right)
calcleafcnt(tree::Leaf{T}) where {T} = 1

builddecisiontriples(leaf::DecisionTree.Leaf{T}, _, _, _, _, _, leaves::Vector{T}, _) where {T} =
    push!(leaves, leaf.majority)
function builddecisiontriples(
    node::DecisionTree.Node{S,T},
    decision_triples::Vector{Tuple{Int,S,Int,NTuple{WSIZE, UInt64}}},
    lo::Int, hi::Int,
    total_size::Int,
    tree_id::Int,
    leaves::Vector{T},
    ::Val{WSIZE},
) where {S,T,WSIZE}
    left_leaf_cnt = calcleafcnt(node.left)
    builddecisiontriples(node.left, decision_triples, lo, lo + left_leaf_cnt - 1, total_size, tree_id, leaves, Val(WSIZE))
    # @assert total_size <= 64
    bv = zeros(UInt64, WSIZE)
    for i = total_size:-1:1
        idx = div(i-1, 64) + 1
        bv[idx] = bv[idx] * 2 + (if lo <= i <= lo + left_leaf_cnt - 1; 0 else 1 end)
    end
    push!(decision_triples, (node.featid, node.featval, tree_id, ntuple(i->bv[i], Val(WSIZE))))
    builddecisiontriples(node.right, decision_triples, lo + left_leaf_cnt, hi, total_size, tree_id, leaves, Val(WSIZE))
end

function QSEnsemble(
    ensemble::DecisionTree.Ensemble{S,T},
    feat_count::Int,
) where {S,T}
    trees = ensemble.trees
    tree_count = length(trees)

    max_leaf_cnt = maximum(calcleafcnt(tree) for tree in trees)
    wsize = div(max_leaf_cnt - 1, 64) + 1

    decision_triples = Vector{Tuple{Int,S,Int,NTuple{wsize, UInt64}}}() # featid, featval, treeid, bv
    leaves = Vector{T}()
    leave_offset = Vector{Int}(undef, tree_count + 1)
    initv = Vector{NTuple{wsize, UInt64}}(undef, tree_count)
    v = Vector{NTuple{wsize, UInt64}}(undef, tree_count)
    leave_offset[1] = 1
    allone = allones(Val(wsize))
    for (treeid, tree) in enumerate(trees)
        leaf_cnt = calcleafcnt(tree)
        builddecisiontriples(tree, decision_triples, 1, leaf_cnt, leaf_cnt, treeid, leaves, Val(wsize))
        leave_offset[treeid + 1] = length(leaves) + 1
        # iv = zero(UInt64); for i = 1:leaf_cnt; iv = iv * 2 + 1; end
        initv[treeid] = allone
    end
    sort!(decision_triples)  ## Q: I wonder how likely it is that different trees have the same triples, in which case we should perhaps keep only one of the copies

    triple_count = length(decision_triples)
    feat_offset = Vector{Int}(undef, feat_count + 1)
    featvals = Vector{S}(undef, triple_count)
    treeids = Vector{Int}(undef, triple_count)
    bvs = Vector{NTuple{wsize,UInt64}}(undef, triple_count)
    vbvs = Vector{NTuple{wsize,UInt64}}(undef, triple_count)
    if triple_count >= 1
        first_offset, featvals[1], treeids[1], bvs[1] = decision_triples[1]
        feat_offset[1:first_offset] .= 1
        vbvs[1] = bvs[1]
        for i = 2:triple_count
            _, featvals[i], treeids[i], bvs[i] = decision_triples[i]
            vbvs[i] = bvs[i]
            if decision_triples[i][1] != decision_triples[i-1][1]
                # in case there are features that don't have corresponding nodes
                for idx=decision_triples[i-1][1]+1:decision_triples[i][1]
                    feat_offset[idx] = i
                end
                # feat_offset[decision_triples[i][1]] = i
            end
        end
        for idx=decision_triples[end][1]+1:feat_count+1
            feat_offset[idx] = triple_count + 1
        end
        # feat_offset[feat_count+1] = triple_count + 1

        for i = 1:feat_count
            # println("$i $(feat_offset[i]) $(feat_offset[i+1]) $feat_count")
            j = feat_offset[i]
            while j + 7 < feat_offset[i + 1]
                for k1 = 0:7, k2 = k1+1:7
                    if treeids[j + k1] == treeids[j + k2]
                        vbvs[j+k1] = vbvs[j+k2] = bvs[j+k1] & bvs[j+k2]
                    end
                end
                j += 8
            end
        end
    end

    QSEnsemble{S,T,wsize}(feat_count, tree_count, leaves, leave_offset, feat_offset, featvals, treeids, bvs, vbvs, initv, v)
end


struct PartialQSEnsemble{S,T,WSIZE}
    qs_ensemble::QSEnsemble{S,T,WSIZE}
    monitored_feats::Vector{Int}
    initv::Vector{NTuple{WSIZE, UInt64}}
    partial_score::Float64
    monitored_trees::Vector{Int}
    # tree_id_to_discretized::Vector{Int}
    # discretized_to_tree_id::Vector{Int}
    # we put v here so we need less frequent gc
    v::Vector{NTuple{WSIZE, UInt64}}
    desired_class::T
end

## Q: We should think about changing the PartialQS so that we can use any changing_set features.
## Alternatively, we would probably need to precompute for PartialQS for different combintations of features for our explanations
@generated function PartialQSEnsemble(qs_ensemble::QSEnsemble{S,T,WSIZE}, feats::Vector{S}, changing_set::AbstractVector{Int}; desired_class=1) where {S,T,WSIZE}
    quote
        _desired_class = convert(T, desired_class)
        @assert qs_ensemble.feat_count == length(feats)
        feat_count = qs_ensemble.feat_count
        tree_count = qs_ensemble.tree_count

        feat_masks = falses(feat_count) # true means changing
        feat_masks[changing_set] .= true

        tree_masks = falses(qs_ensemble.tree_count)
        pre_initv = copy(qs_ensemble.initv)
        for (k, (mask, feat)) in enumerate(zip(feat_masks, feats))
            i = qs_ensemble.feat_offset[k]
            r = qs_ensemble.feat_offset[k + 1]
            if !mask # this feature is fixed
                while i < r && feat >= qs_ensemble.featvals[i]
                    h = qs_ensemble.treeids[i]
                    pre_initv[h] &= qs_ensemble.bvs[i]
                    i += 1
                end
            else # this feature is changing - mark all treeid influenced
                for idx = i:r-1
                    tree_masks[qs_ensemble.treeids[idx]] = true
                end
            end
        end

        monitored_feats = changing_set
        monitored_trees = Vector{Int}()

        initv = pre_initv
        # initv = Vector{UInt64}()
        # tree_id_to_discretized = Vector{Int}(undef, tree_count)
        # discretized_to_tree_id = Vector{Int}()
        partial_score = zero(
            $(if T <: Float64
                :T
            else
                :Int
            end))
        for h = 1:tree_count
            if !tree_masks[h] # this tree will not be updated, calculate it now
                j = trailing_zeros(pre_initv[h])
                offset = qs_ensemble.leave_offset[h] + j
                res = qs_ensemble.leaves[offset]

                $(if T <: Float64
                    :(@fastmath partial_score += res)
                else
                    :(partial_score += res == _desired_class)
                end)
            else # this tree is changing
                push!(monitored_trees, h)
            end
        end

        v = Vector{NTuple{WSIZE, UInt64}}(undef, length(initv))
        PartialQSEnsemble(qs_ensemble, monitored_feats, initv, Float64(partial_score),
            # tree_id_to_discretized, discretized_to_tree_id,
            monitored_trees, v, _desired_class)
    end
end

function barrierhelper(feats::Vector{S2}, k, v, qs_ensemble::QSEnsemble{S,T}) where {S,T,S2}
    for ent = 1:length(feats)
        feat::S = convert(S, feats[ent])
        i::Int = qs_ensemble.feat_offset[k]
        r::Int = qs_ensemble.feat_offset[k + 1]
        # no vectorization currently because
        # while i + 7 < r && feat >= qs_ensemble.featvals[i + 7]
        #     hs = vload(Vec{8,Int}, qs_ensemble.treeids, i)
        #     bvs = vload(Vec{8,UInt64}, qs_ensemble.vbvs, i)
        #     vscatter(vgather(v, hs) & bvs, v, hs)
        #     i += 8
        # end

        while i < r && feat >= qs_ensemble.featvals[i]
            h0 = qs_ensemble.treeids[i]

            v[h0, ent] &= qs_ensemble.bvs[i]
            i += 1
        end
    end
end


@generated function eval_partial_ensemble(p_ensemble::PartialQSEnsemble{S,T}, changing_feats::AbstractVector{S}) where {S,T}
    quote
        @assert length(changing_feats) >= length(p_ensemble.monitored_feats)
        qs_ensemble = p_ensemble.qs_ensemble
        v = p_ensemble.v
        v .= p_ensemble.initv
        # discretized_tree_count = length(p_ensemble.initv)
        tree_count = length(p_ensemble.initv)
        feat_count = length(p_ensemble.monitored_feats)

        for idx = 1:feat_count
            k = p_ensemble.monitored_feats[idx]
            feat = changing_feats[idx]
            i = qs_ensemble.feat_offset[k]
            r = qs_ensemble.feat_offset[k + 1]
            # while i + 7 < r && feat >= qs_ensemble.featvals[i + 7]
            #     hs = vload(Vec{8,Int}, qs_ensemble.treeids, i)
            #     bvs = vload(Vec{8,UInt64}, qs_ensemble.vbvs, i)
            #     vscatter(vgather(v, hs) & bvs, v, hs)
            #     i += 8
            # end

            while i < r && feat >= qs_ensemble.featvals[i]
                h0 = qs_ensemble.treeids[i]

                v[h0] &= qs_ensemble.bvs[i]
                i += 1
            end
        end

        # score = p_ensemble.partial_score
        score = zero(
            $(if T <: Float64
                :T
            else
                :Int
            end))

        # TODO: optimize here
        begin
            monitored_tree_count = length(p_ensemble.monitored_trees)
            i = 1
            while i <= monitored_tree_count
                h = p_ensemble.monitored_trees[i]
                j = trailing_zeros(v[h])
                offset = qs_ensemble.leave_offset[h] + j
                res = qs_ensemble.leaves[offset]
                $(if T <: Float64
                    :(@fastmath score += res)
                else
                    :(score += res == p_ensemble.desired_class)
                end)
                i += 1
            end
        end
        (p_ensemble.partial_score + score) / qs_ensemble.tree_count
    end
end


@generated function eval_partial_ensemble(p_ensemble::PartialQSEnsemble{S,T}, df::DataFrame) where {S,T}
    quote
        # println("$(ncol(df)) $(length(p_ensemble.monitored_feats))")
        # println(names(df))
        # @assert ncol(df) == length(p_ensemble.monitored_feats) + 1
        @assert ncol(df) == length(p_ensemble.monitored_feats) + 3

        qs_ensemble = p_ensemble.qs_ensemble
        v = repeat(p_ensemble.initv, 1, nrow(df))
        # discretized_tree_count = length(p_ensemble.initv)
        tree_count = length(p_ensemble.initv)
        feat_count = length(p_ensemble.monitored_feats)

        for idx = 1:feat_count
            feats = df[:, idx]
            k = p_ensemble.monitored_feats[idx]

            barrierhelper(feats, k, v, qs_ensemble)
        end

        # score = p_ensemble.partial_score
        scores = zeros(
            $(if T <: Float64
                :T
            else
                :Int
            end), nrow(df))

        # TODO: optimize here
        begin
            monitored_tree_count = length(p_ensemble.monitored_trees)

            for ent = 1 : nrow(df)
                i = 1
                while i <= monitored_tree_count
                    h = p_ensemble.monitored_trees[i]
                    j = trailing_zeros(v[h, ent])
                    offset = qs_ensemble.leave_offset[h] + j
                    res = qs_ensemble.leaves[offset]
                    $(if T <: Float64
                        :(@fastmath scores[ent] += res)
                    else
                        :(scores[ent] += res == p_ensemble.desired_class)
                    end)
                    i += 1
                end
            end
        end
        return (p_ensemble.partial_score .+ scores) ./ qs_ensemble.tree_count
    end
end

@generated function eval_ensemble(ensemble::QSEnsemble{S,T}, feats::Vector{S}; desired_class = 0) where {S,T}
    quote
        _desired_class = convert(T, desired_class)
        v = ensemble.v
        v .= ensemble.initv

        begin
            k = 1
            while k <= ensemble.feat_count
                feat = feats[k]
                i = ensemble.feat_offset[k]
                r = ensemble.feat_offset[k + 1]
                while i < r && feat >= ensemble.featvals[i]
                    h = ensemble.treeids[i]
                    v[h] &= ensemble.bvs[i]
                    i += 1
                end
                k += 1
            end
        end

        score = zero(
            $(if T <: Float64
                :T
            else
                :Int
            end))
        h = 1
        while h <= ensemble.tree_count
            j = trailing_zeros(v[h])
            offset = ensemble.leave_offset[h] + j
            res = ensemble.leaves[offset]
            $(if T <: Float64
                :(@fastmath score += res)
            else
                :(score += res == _desired_class)
            end)
            h += 1
        end
        Float64(score) / ensemble.tree_count
    end
end

@generated function eval_ensemble(ensemble::QSEnsemble{S,T}, df::DataFrame; desired_class = 0) where {S,T}
    quote
        _desired_class = convert(T, desired_class)
        v = repeat(ensemble.initv, 1, nrow(df))

        for k = 1:ensemble.feat_count
            feats = df[:, k]

            barrierhelper(feats, k, v, ensemble)
        end

        scores = zeros(
            $(if T <: Float64
                :T
            else
                :Int
            end), nrow(df))
        for ent = 1:nrow(df)
            h = 1
            while h <= ensemble.tree_count
                j = trailing_zeros(v[h, ent])
                offset = ensemble.leave_offset[h] + j
                res = ensemble.leaves[offset]
                $(if T <: Float64
                    :(@fastmath scores[ent] += res)
                else
                    :(scores[ent] += res == _desired_class)
                end)
                h += 1
            end
        end
        scores ./= ensemble.tree_count
        return scores
    end
end

function randomtree(depth, featid_range, featval_range, majority_range, ::Type{T}) where {T}
    function randomtreeimpl(depth)
        if depth > 0
            featid = rand(featid_range)
            featval = rand(featval_range)
            DecisionTree.Node(
                featid,
                featval,
                randomtreeimpl(depth - 1),
                randomtreeimpl(depth - 1),
            )
        else
            DecisionTree.Leaf{T}(rand(majority_range), T[])
        end
    end
    randomtreeimpl(depth)
end

end


# # demo code below
# qs = QuickScorer
# using BenchmarkTools,DecisionTree
# using SIMD

# forest = collect(DecisionTree.LeafOrNode{Int,Int}, [qs.randomtree(rand(4:6), 1:200, 1:1000, 1:2, Int) for i = 1:1000])
# ensemble = DecisionTree.Ensemble(forest)
# qsensemble = qs.QSEnsemble(ensemble, 200)
# feat = [rand(1:1000) for i = 1:200]
# pensemble = QuickScorer.PartialQSEnsemble(qsensemble,feat,collect(10:10:200), desired_class = 1)
# # pensemble = QuickScorer.PartialQSEnsemble(qsensemble,feat,Int[])
# pfeat = feat[10:10:200]

# @profiler for i = 1:100000
#     qs.eval_partial_ensemble(pensemble, pfeat)
#     # qs.eval_ensemble(qsensemble, feat)
# end
# using Traceur
# apply_forest(ensemble, feat)
# QuickScorer.eval_ensemble(qsensemble, feat, desired_class = 1)
# QuickScorer.eval_partial_ensemble(pensemble, pfeat)
# x=[apply_tree(tree, feat) for tree in ensemble.trees]
# count(==(1), x)
# QuickScorer.eval_partial_ensemble(pensemble, Int[])


# @benchmark apply_forest(ensemble, feat)
# @benchmark QuickScorer.eval_ensemble(qsensemble, feat)
# @benchmark QuickScorer.eval_partial_ensemble(pensemble, pfeat)
# @profiler for i=1:100000
#     QuickScorer.eval_partial_ensemble(pensemble, pfeat)
# end
# x=Vec{4,Int64}((1,2,3,4))
# x+x
#
# @fastmath x+x
