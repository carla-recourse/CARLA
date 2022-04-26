module RapidScorer

using MLJModels, MLJ
using DecisionTree
using DataFrames
using SIMD

## Epitome ##

struct EpitomeWrapper{EpitomeType,S}
    featids::Vector{Int} # compressed feature ids (has the same length as feat_offset)
    feat_offset::Vector{Int} # offset vector into featvals and epitome_offset
    featvals::Vector{S}
    epitome_offset::Vector{Int} # offset vector into treeids and epitomes
    treeids::Vector{Int}
    epitomes::Vector{EpitomeType}
    function EpitomeWrapper(::Type{EpitomeType}, ::Type{S}) where {EpitomeType,S}
        new{EpitomeType, S}(Int[], Int[], S[], Int[], Int[], EpitomeType[])
    end
end

function Base.push!(
    ew::EpitomeWrapper{EpitomeType, S},
    triple::Tuple{Int, S,Int,EpitomeType}
) where {EpitomeType,S}
    featid, featval, treeid, epitome = triple
    push!(ew.treeids, treeid)
    push!(ew.epitomes, epitome)
    if length(ew.featvals) == 0 || featval != ew.featvals[end]
        push!(ew.featvals, featval)
        push!(ew.epitome_offset, length(ew.epitomes))
    end
    if length(ew.featids) == 0 || featid != ew.featids[end]
        push!(ew.featids, featid)
        push!(ew.feat_offset, length(ew.featvals))
    end
end

function addoveroffset!(ew::EpitomeWrapper{EpitomeType, S}) where {EpitomeType, S}
    push!(ew.epitome_offset, length(ew.epitomes) + 1)
    push!(ew.feat_offset, length(ew.featvals) + 1)
end

struct Epitome
    fb :: UInt8; fbp :: Int
    eb :: UInt8; ebp :: Int
end

struct ShortEpitome
    fb :: UInt8; fbp :: Int
end

## QSData ##

# NSIZE is the number of 64bits for the width of trees
# VSIZE is the number of samples run in parallel - we need it to adjust the v_bitvectors
# for soundness
struct QSDataWrapper{NSIZE, VSIZE, S}
    featids::Vector{Int} # compressed feature ids (has the same length as feat_offset)
    feat_offset::Vector{Int} # offset vector into featvals and epitome_offset
    featvals::Vector{S}
    treeids::Vector{Int}
    bitvectors::Vector{UInt64}
    v_bitvectors::Vector{UInt64}

    function QSDataWrapper(::Val{NSIZE}, ::Val{VSIZE}, ::Type{S}) where {NSIZE,VSIZE,S}
        new{NSIZE, VSIZE, S}(Int[], Int[], S[], Int[], UInt64[], UInt64[])
    end
end

function Base.push!(
    ew::QSDataWrapper{NSIZE, VSIZE, S},
    triple::Tuple{Int, S,Int,Vec{NSIZE,UInt64}}
) where {NSIZE,VSIZE,S}
    featid, featval, treeid, bitvector = triple
    push!(ew.treeids, treeid)
    for i = 1:NSIZE
        push!(ew.bitvectors, bitvector[i])
        push!(ew.v_bitvectors, bitvector[i])
    end
    push!(ew.featvals, featval)
    if length(ew.featids) == 0 || featid != ew.featids[end]
        push!(ew.featids, featid)
        push!(ew.feat_offset, length(ew.featvals))
    end
end

function addoveroffset!(ew::QSDataWrapper{NSIZE, VSIZE, S}) where {NSIZE, VSIZE, S}
    push!(ew.feat_offset, length(ew.featvals) + 1)

    # besides adding the explicit over-offset, this method will also
    # create the v_bitvectors array

    vbv = ew.v_bitvectors
    for i = 1:length(ew.featids)
        j = ew.feat_offset[i]
        while j + 8 <= ew.feat_offset[i + 1]
            for k1 = 0:7, k2 = k1+1:7
                if ew.treeids[j + k1] == ew.treeids[j + k2]
                    for l = 1:NSIZE
                        vbv[(j+k1-1)*NSIZE+l] = vbv[(j+k2-1)*NSIZE+l] = vbv[(j+k1-1)*NSIZE+l] & vbv[(j+k2-1)*NSIZE+l]
                    end
                end
            end
            j += 8
        end
    end
end

## Ensemble ##

# S is the type of feature value for comparison, T is the type of leaf value
# VSIZE is the size of feature vectors calculated in parallel,
# MSIZE is the size of mask vector for each tree (= LAMBDA / 8)
# USIZE should always be VSIZE / 8, it's put here because we want to prompt
# the compiler that this is a known value during compile time
struct RSEnsemble{S,T,VSIZE,MSIZE,USIZE,NSIZE}
    tree_count::Int
    leaf_offset::Vector{Int}
    leaves::Vector{T}

    qs_data::QSDataWrapper{NSIZE,VSIZE,S}

    epitomes::EpitomeWrapper{Epitome,S}
    short_epitomes::EpitomeWrapper{ShortEpitome,S}

    # we put v here so we need less frequent gc
    # initv is used for preprocessing a la QS
    initv::Vector{UInt64}
    v::BitVector
    int_score::Vector{Int}
end

calcleafcnt(tree::DecisionTree.Node{S,T}) where {S,T} =
    calcleafcnt(tree.left) + calcleafcnt(tree.right)
calcleafcnt(tree::Leaf{T}) where {T} = 1

function buildvecfromepitome(fb, fbp, eb, ebp, ::Val{NSIZE}) where {NSIZE}
    #TODO: performance could be improved
    ntuple(i -> begin
        val = UInt64(0)
        for j = 7:-1:0
            pos = (i - 1) * 8 + j
            val = val << 8 + (if pos < fbp || pos > ebp; 0xff
                              elseif pos == fbp && pos == ebp; fb | eb
                              elseif pos == fbp; fb
                              elseif pos == ebp; eb
                              else 0x00 end)
        end
        val
    end, Val(NSIZE)) |> Vec
end

# lo and hi are left-inclusive, right-exclusive
builddecisiontriples(leaf::DecisionTree.Leaf{T}, _, _, _, _, leaves::Vector{T},_) where {T} =
    push!(leaves, leaf.majority)
function builddecisiontriples(
    node::DecisionTree.Node{S,T},
    decision_triples::Vector{Tuple{Int,S,Int,Epitome, Vec{NSIZE,UInt64}}},
    lo::Int, hi::Int,
    tree_id::Int,
    leaves::Vector{T},
    valnsize::Val{NSIZE}
) where {S,T,NSIZE}
    left_leaf_cnt = calcleafcnt(node.left)
    builddecisiontriples(node.left, decision_triples, lo, lo + left_leaf_cnt, tree_id, leaves, valnsize)

    # divrem(lo, 8)
    fbp, fbb = lo >> 3, lo & 7
    # should minus one here since the range is [lo, hi)
    ebp, ebb = (lo + left_leaf_cnt - 1) >> 3, (lo + left_leaf_cnt - 1) & 7

    fb = 0x0
    for i = 7:-1:0
        # consider fbb to be 2 for example, then fb shall be 00000011
        fb = fb * 2 + (if i >= fbb; 0 else 1 end)
    end
    eb = 0x0
    for i = 7:-1:0
        # consider ebb to be 2 for example, then eb shall be 11111000
        eb = eb * 2 + (if i <= ebb; 0 else 1 end)
    end

    vec = buildvecfromepitome(fb, fbp, eb, ebp, Val(NSIZE))

    push!(decision_triples, (node.featid, node.featval, tree_id, Epitome(fb, fbp, eb, ebp), vec))
    builddecisiontriples(node.right, decision_triples, lo + left_leaf_cnt, hi, tree_id, leaves, valnsize)
end

function RSEnsemble(
    ensemble::DecisionTree.Ensemble{S,T},
    ::Val{VSIZE}, ::Val{USIZE}
) where {S,T,VSIZE,USIZE}
    # MSIZE = (div(maximum(calcleafcnt(tree) for tree in ensemble.trees) - 1, 64) + 1) * 64
    MSIZE = (div(maximum(calcleafcnt(tree) for tree in ensemble.trees) - 1, 8) + 1)
    trees = ensemble.trees
    tree_count = length(trees)

    decision_triples = Vector{Tuple{Int,S,Int,Epitome,Vec{MSIZE>>3, UInt64}}}() # featid, featval, treeid, epitome, vec
    leaves = Vector{T}()
    leaf_offset = Vector{Int}(undef, tree_count + 1)
    leaf_offset[1] = 1
    for (treeid, tree) in enumerate(trees)
        leaf_cnt = calcleafcnt(tree)
        builddecisiontriples(tree, decision_triples, 0, leaf_cnt, treeid, leaves, Val(MSIZE>>3))
        leaf_offset[treeid + 1] = length(leaves) + 1
    end
    sort!(decision_triples, by = triple -> triple[1:2]) # I guess this may be slightly inefficient

    qs_data = QSDataWrapper(Val(MSIZE>>3), Val(VSIZE), S)

    epitomes = EpitomeWrapper(Epitome, S)
    short_epitomes = EpitomeWrapper(ShortEpitome, S)

    for triple in decision_triples
        featid, featval, treeid, epitome, vec = triple
        if epitome.fbp == epitome.ebp # where the epitome should go
            short_epitome = ShortEpitome(epitome.fb | epitome.eb, epitome.fbp)
            push!(short_epitomes, (featid, featval, treeid, short_epitome))
        else
            push!(epitomes, (featid, featval, treeid, epitome))
        end
        push!(qs_data, (featid, featval, treeid, vec))
    end
    addoveroffset!(epitomes)
    addoveroffset!(short_epitomes)
    addoveroffset!(qs_data)

    v = BitVector(undef, 8 * tree_count * VSIZE * MSIZE)
    initv = Vector{UInt64}(undef, tree_count * MSIZE >> 3)
    int_score = Vector{Int}(undef, VSIZE)

    RSEnsemble{S,T,VSIZE,MSIZE,USIZE,MSIZE>>3}(tree_count, leaf_offset, leaves, qs_data, epitomes, short_epitomes, initv, v, int_score)
end

@generated function preprocessepitomes(
    ew::QSDataWrapper{NSIZE,VSIZE,S}, template_feats::Vector{S},
    changing_set::AbstractVector{Bool},
    tree_count::Int,
    initv::Vector{UInt64}, v::BitVector,
    ::Val{VSIZE}, ::Val{MSIZE}, ::Val{USIZE}
) where {S,VSIZE,MSIZE,USIZE,NSIZE}
    # quote @inbounds begin
    quote begin
        feat_count = length(ew.featids)
        initv .= ~UInt(0)
        k = 1
        while k <= feat_count
            if !changing_set[ew.featids[k]]
                feat = template_feats[ew.featids[k]]
                i = ew.feat_offset[k]
                r = ew.feat_offset[k + 1]
                # here we are "abusing" USIZE a little bit
                # Also note here we are using v_bitvectors instead of bitvectors to
                # avoid conflicts
                #
                # UPDATE: don't do this, as it may break locality
                #
                # while i + USIZE - 1 < r && feat >= ew.featvals[i + USIZE - 1]
                #     h = ew.treeids[VecRange{USIZE}(i)]
                #     initvidx = (h - 1) * NSIZE
                #     bvidx = (i - 1 + Vec{USIZE, Int64}(($((0:USIZE-1)...),))) * NSIZE
                #     $(begin
                #         expr = quote end
                #         for npos = 1:NSIZE
                #             expr = quote
                #                 $expr
                #                 initv[initvidx + $npos] &= ew.v_bitvectors[bvidx + $npos]
                #             end
                #         end
                #         expr
                #     end)
                #     i += USIZE
                # end
                while i < r && feat >= ew.featvals[i]
                    h = ew.treeids[i]
                    initvidx = (h - 1) * NSIZE
                    bvidx = (i - 1) * NSIZE
                    $(begin
                        expr = quote end
                        for npos = 1:NSIZE
                            expr = quote
                                $expr
                                initv[initvidx + $npos] &= ew.bitvectors[bvidx + $npos]
                            end
                        end
                        expr
                    end)
                    i += 1
                end
            end
            k += 1
        end

        vchunks = v.chunks
        lane = VecRange{USIZE}(0)
        for i = 0:tree_count - 1
            for j = 0:NSIZE - 1
                for k = 0:7
                    # val = reinterpret(UInt8, initv)[i * NSIZE * 8 + j * 8 + k + 1]
                    val = UInt8((initv[i * NSIZE + j + 1] >> (8 * k)) & UInt(0xff))
                    vchunks[i * MSIZE * USIZE + j * 8 * USIZE + k * USIZE + 1 + lane] = reinterpret(Vec{USIZE, UInt64}, Vec{VSIZE, UInt8}(val))
                end
            end
        end
    end end
end

@inline @generated function processepitomes(
    ew::EpitomeWrapper{EpitomeType,S}, compressed_feats::Vector{S},
    real_to_compressed_feats_map::Vector{Int}, changing_set::AbstractVector{Bool},
    v::BitVector, ::Val{VSIZE}, valmsize::Val{MSIZE}, valusize::Val{USIZE}
) where {EpitomeType<:Union{Epitome, ShortEpitome},S,VSIZE, MSIZE,USIZE}
    vectorized_and = if EpitomeType <: Epitome
        quote
            chunks = v.chunks
            base = (h - 1) * USIZE * MSIZE + 1

            val = reinterpret(Vec{USIZE, UInt64}, mask | Vec{VSIZE, UInt8}(p.fb))
            offset = base + p.fbp * USIZE
            # @inbounds vstore(vload(Vec{USIZE,UInt64}, chunks, offset) & val, chunks, offset)
            vstore(vload(Vec{USIZE,UInt64}, chunks, offset) & val, chunks, offset)

            for kk = p.fbp + 1 : p.ebp - 1
                val = reinterpret(Vec{USIZE, UInt64}, mask)
                offset = base + kk * USIZE
                # @inbounds vstore(vload(Vec{USIZE,UInt64}, chunks, offset) & val, chunks, offset)
                vstore(vload(Vec{USIZE,UInt64}, chunks, offset) & val, chunks, offset)
            end

            val = reinterpret(Vec{USIZE, UInt64}, mask | Vec{VSIZE, UInt8}(p.eb))
            offset = base + p.ebp * USIZE
            # @inbounds vstore(vload(Vec{USIZE,UInt64}, chunks, offset) & val, chunks, offset)
            vstore(vload(Vec{USIZE,UInt64}, chunks, offset) & val, chunks, offset)
        end
    else
        quote
            chunks = v.chunks::Vector{UInt64}
            base = (h - 1) * USIZE * MSIZE + 1
            val = reinterpret(Vec{USIZE, UInt64}, mask | Vec{VSIZE, UInt8}(p.fb))
            offset = base + p.fbp * USIZE
            # @inbounds vstore(vload(Vec{USIZE,UInt64}, chunks, offset) & val, chunks, offset)
            vstore(vload(Vec{USIZE,UInt64}, chunks, offset) & val, chunks, offset)
        end
    end
    # quote @inbounds begin
    quote begin
        feat_count = length(ew.featids)
        lane = VecRange{VSIZE}(0)
        allones = Vec{VSIZE,UInt8}(0xff)
        allzeros = Vec{VSIZE,UInt8}(0)
        k = 1
        while k <= feat_count
            if changing_set[ew.featids[k]]
                xk = compressed_feats[(real_to_compressed_feats_map[ew.featids[k]] - 1) * VSIZE + 1 + lane]
                i = ew.feat_offset[k]
                while i < ew.feat_offset[k + 1]
                    mask = vifelse(xk < ew.featvals[i], allones, allzeros)
                    # using UInt64 comparison for speedup
                    if any(reinterpret(Vec{USIZE, UInt64}, mask) != reinterpret(Vec{USIZE,UInt64}, allones))
                        q = ew.epitome_offset[i]
                        while q <= ew.epitome_offset[i + 1] - 1
                            h = ew.treeids[q]
                            p = ew.epitomes[q]
                            $vectorized_and
                            q += 1
                        end
                    else
                        break
                    end
                    i += 1
                end
            end
            k += 1
        end
    end end
end

function eval_ensemble(
    ensemble::RSEnsemble{S,T,VSIZE,MSIZE,USIZE,NSIZE},
    compressed_feats::Vector{S},
    real_to_compressed_feats_map::Vector{Int},
    changing_set::AbstractVector{Bool},
    template_feats::Vector{S};
    desired_class = 0
) where {S,T,VSIZE,MSIZE,USIZE,NSIZE}
    score = Vector{Float64}(undef, VSIZE)
    eval_ensemble(score,
        ensemble, compressed_feats, real_to_compressed_feats_map,
        changing_set, template_feats, desired_class = desired_class)
end

@generated function eval_ensemble(
    score::Vector{Float64},
    ensemble::RSEnsemble{S,T,VSIZE,MSIZE,USIZE,NSIZE},
    compressed_feats::Vector{S},
    real_to_compressed_feats_map::Vector{Int},
    changing_set::AbstractVector{Bool},
    template_feats::Vector{S};
    desired_class = 0
) where {RT,S,T,VSIZE,MSIZE,USIZE,NSIZE}
    # quote @inbounds begin
    quote begin
        $(if !(T <: Float64)
            :(ensemble.int_score .= 0)
        end)
        _desired_class = convert(T, desired_class)
        v = ensemble.v
        vchunks = v.chunks
        fill!(vchunks, ~UInt(0))

        preprocessepitomes(ensemble.qs_data, template_feats, changing_set, ensemble.tree_count, ensemble.initv, v, Val(VSIZE), Val(MSIZE), Val(USIZE))

        processepitomes(ensemble.epitomes, compressed_feats,
                        real_to_compressed_feats_map, changing_set,
                        v, Val(VSIZE), Val(MSIZE), Val(USIZE))
        processepitomes(ensemble.short_epitomes, compressed_feats,
                        real_to_compressed_feats_map, changing_set,
                        v, Val(VSIZE), Val(MSIZE), Val(USIZE))

        lane = VecRange{USIZE}(0)
        h = 1
        while h <= ensemble.tree_count
            base = (h - 1) * USIZE * MSIZE + 1
            byte_pos = Vec{VSIZE, UInt8}(0)
            byte_val = Vec{VSIZE, UInt8}(0)
            # we update from the last to the first, whenever we see a nonzero v-value,
            # we overwrite it, so we can ensure byte_val is the leftest byte and byte_pos
            # is the leftest byte's position

            $(begin
                expr = quote end
                for i = MSIZE - 1:-1:0
                    expr = quote
                        $expr
                        val = reinterpret(Vec{VSIZE, UInt8}, vchunks[base + $(i * USIZE) + lane])
                        byte_pos = vifelse(val != 0, Vec{VSIZE, UInt8}(UInt8($i)), byte_pos)
                        byte_val = vifelse(val != 0, val, byte_val)
                    end
                end
                expr
            end)


            bit_pos = Vec{VSIZE, UInt8}(0)
            $(begin
                expr = quote end
                for i = 7:-1:0
                    expr = quote
                        $expr
                        mask = UInt8($(1 << i))
                        bit_pos = vifelse((byte_val & mask) != 0, Vec{VSIZE,UInt8}(UInt8($i)), bit_pos)
                    end
                end
                expr
            end)


            $(begin
                expr = quote end
                for i = 1:VSIZE
                    expr = quote
                        $expr
                        pos = Int(byte_pos[$i]) << 3 + bit_pos[$i]
                        offset = ensemble.leaf_offset[h] + pos
                        $(if T <: Float64
                            :(@fastmath score[$i] += ensemble.leaves[offset])
                        else
                            :(ensemble.int_score[$i] += ensemble.leaves[offset] == _desired_class)
                        end)
                    end
                end
                expr
            end)
            h += 1
        end
        $(begin
            if T <: Float64
                :(score ./= ensemble.tree_count)
            else
                :(score .= ensemble.int_score ./ ensemble.tree_count)
            end
        end)
    end end
end

end

# demo below

using BenchmarkTools,DecisionTree

function randomtree(depth, featid_range, featval_range, majority_range)
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
            DecisionTree.Leaf{Float64}(rand(majority_range), Float64[])
        end
    end
    randomtreeimpl(depth)
end
using Test
function test_quickscorer_implementation()
    forest = collect(DecisionTree.LeafOrNode{Int,Float64}, [randomtree(rand(4:6), 1:200, 1:1000, 0.1:0.05:1) for i = 1:1000])
    ensemble = DecisionTree.Ensemble(forest)
    rsensemble = RapidScorer.RSEnsemble(ensemble, Val(32), Val(4))

    feat = [rand(1:1000) for i = 1:200*32]
    rs_result = RapidScorer.eval_ensemble(rsensemble, feat, collect(1:200), trues(200), Int[])

    ground_truth = [apply_forest(ensemble, feat[i:32:200*32]) for i in 1:32]
    @assert isapprox(rs_result, ground_truth)
end

function test_rapidscorer_implementation()
    forest = collect(DecisionTree.LeafOrNode{Int,Float64}, [randomtree(rand(4:6), 1:200, 1:1000, 0.1:0.05:1) for i = 1:1000])
    ensemble = DecisionTree.Ensemble(forest)
    rsensemble = RapidScorer.RSEnsemble(ensemble, Val(32), Val(4))

    template_feats = [rand(1:1000) for i = 1:200]
    changing_set = [i % 20 == 1 for i = 1:200]
    real_to_compressed_feats_map = [(if i % 20 == 1; div(i, 20) + 1 else 0 end) for i = 1:200]
    compressed_feats = [rand(1:1000) for i = 1:10*32]

    rs_result_1 = RapidScorer.eval_ensemble(rsensemble, Int[], zeros(Int, 200), falses(200), template_feats)
    ground_truth_1 = [apply_forest(ensemble, template_feats) for _=1:32]
    @assert isapprox(rs_result_1, ground_truth_1)

    rs_result_2 = RapidScorer.eval_ensemble(rsensemble, compressed_feats, real_to_compressed_feats_map, changing_set, template_feats)
    ground_truth_2 = [
        apply_forest(ensemble, [
            (if i % 20 == 1
                compressed_feats[(real_to_compressed_feats_map[i] - 1) * 32 + j]
            else
                template_feats[i]
            end)
            for i = 1:200
        ]) for j = 1:32]
    @assert isapprox(rs_result_2, ground_truth_2)
end



# end

# forest = collect(DecisionTree.LeafOrNode{Int,Float64}, [randomtree(rand(4:6), 1:200, 1:1000, 0.1:0.05:1) for i = 1:1000])
# ensemble = DecisionTree.Ensemble(forest)
# rsensemble = RapidScorer.RSEnsemble(ensemble, Val(32), Val(4))
# rsensemble = RapidScorer.RSEnsemble(ensemble, Val(64), Val(8))
#
# template_feats = [rand(1:1000) for i = 1:200]
# changing_set = [i % 20 == 1 for i = 1:200]
# real_to_compressed_feats_map = [(if i % 20 == 1; div(i, 20) + 1 else 0 end) for i = 1:200]
# compressed_feats = [rand(1:1000) for i = 1:10*32]
# compressed_feats = [rand(1:1000) for i = 1:10*64]

# res = RapidScorer.eval_ensemble(rsensemble, compressed_feats, real_to_compressed_feats_map, changing_set, template_feats)

# @benchmark begin
#     RapidScorer.eval_ensemble(rsensemble, compressed_feats, real_to_compressed_feats_map, changing_set, template_feats)
# end

# ground_truth_2 = [
#     apply_forest(ensemble, [
#         (if i % 20 == 1
#             compressed_feats[(real_to_compressed_feats_map[i] - 1) * 32 + j]
#         else
#             template_feats[i]
#         end)
#         for i = 1:200
#     ]) for j = 1:32]

# @benchmark [apply_forest(ensemble, feat[i:32:200*32]) for i in 1:32]
# @benchmark RapidScorer.eval_ensemble(rsensemble, feat)
# @benchmark RapidScorer.eval_ensemble(rsensemble, feat, collect(1:1000), trues(1000), Int[])
# @profiler for i = 1:10000
#     RapidScorer.eval_ensemble(rsensemble, compressed_feats, real_to_compressed_feats_map, changing_set, template_feats)
# end
# @code_warntype RapidScorer.eval_ensemble(rsensemble, feat)
