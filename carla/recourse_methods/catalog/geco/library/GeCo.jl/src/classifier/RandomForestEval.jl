module RandomForestEvaluation

export RandomForestEval, PartialRandomForestEval,
       initPartialRandomForestEval, initRandomForestEval,
       predict

using MLJ, MLJModels, DataFrames
# using MLJ, MLJModels, DecisionTree, DataFrames

include("RapidScorer.jl")
using .RapidScorer
include("QuickScorer.jl")
using .QuickScorer

const NUM_EXTRA_COL = 4
const NUM_EXTRA_FEASIBLE_SPACE_COL = 2

struct RandomForestEval{S,T,VSIZE,MSIZE,USIZE,NSIZE}
    rsensemble::RapidScorer.RSEnsemble{S,T,VSIZE,MSIZE,USIZE,NSIZE}
    qsensemble::QuickScorer.QSEnsemble{S,T}
    desired_value::T
    orig_instance::Vector{Float64}
end

# TODO add WSIZE
struct PartialRandomForestEval{S,T,VSIZE,MSIZE,USIZE,NSIZE,WSIZE}
    rsensemble::RapidScorer.RSEnsemble{S,T,VSIZE,MSIZE,USIZE,NSIZE}
    qsensemble::QuickScorer.QSEnsemble{S,T,WSIZE}
    desired_value::T
    partial_cache::Dict{BitVector, QuickScorer.PartialQSEnsemble{S, T}}
    orig_instance::Vector{Float64}
end

initPartialRandomForestEval(classifier, orig_instance, desired_class) = begin
    # lambda = nextpow(2, maximum(RapidScorer.calcleafcnt.(ensemble.trees)))
    ensemble = classifier.fitresult[1]
    lambda = 64
    num_feat = size(orig_instance, 1)

    desired_class_value = classifier.fitresult[3][findfirst(isequal(desired_class), classifier.fitresult[2])]

    rsensemble = RapidScorer.RSEnsemble(ensemble, Val(32), Val(4))
    qsensemble = QuickScorer.QSEnsemble(ensemble, num_feat)

    PartialRandomForestEval(
        rsensemble,
        qsensemble,
        desired_class_value,
        Dict{BitVector, QuickScorer.PartialQSEnsemble{Float64, UInt32}}(),
        collect(Float64, orig_instance))
end

initRandomForestEval(classifier, orig_instance, desired_class) = begin
    ensemble = classifier.fitresult[1]
    num_feat = size(orig_instance, 1)

    desired_class_value = classifier.fitresult[3][findfirst(isequal(desired_class), classifier.fitresult[2])]

    rsensemble = RapidScorer.RSEnsemble(ensemble, Val(32), Val(4))
    qsensemble = QuickScorer.QSEnsemble(ensemble, num_feat)

    RandomForestEval(
        rsensemble,
        qsensemble,
        desired_class_value,
        collect(Float64, orig_instance))
end

# column style, partial entity, partial q/rs
@inline function predict(ensemble::PartialRandomForestEval, df::DataFrame, mod::BitVector)
    # orig_instance_df = manager.orig_instance_df
    # max_num_entity = 0
    # for (mod, entities) in manager.dict
    #     max_num_entity = max(nrow(entities), max_num_entity)
    # end

    partialqsensemble = if haskey(ensemble.partial_cache, mod)
        ensemble.partial_cache[mod]
    else
        rsensemble = ensemble.rsensemble
        qsensemble = ensemble.qsensemble

        desired_class_value =  ensemble.desired_value
        orig_instance = ensemble.orig_instance

        qs_changing_set = [i for (i, b) in enumerate(mod) if b]
        ensemble.partial_cache[mod] = QuickScorer.PartialQSEnsemble(qsensemble, orig_instance, qs_changing_set, desired_class = desired_class_value)
    end

    pred::Vector{Float64} = QuickScorer.eval_partial_ensemble(partialqsensemble, df)

    return pred
end

# row style, full entity partial q/rs
@inline function predict(ensemble::PartialRandomForestEval, population::DataFrame)
    cache = ensemble.partial_cache
    rsensemble = ensemble.rsensemble
    qsensemble = ensemble.qsensemble
    orig_instance = ensemble.orig_instance
    # fitresult = ensemble.fitresult
    desired_class_value = ensemble.desired_value

    grouped_entities = groupby(population, :mod)
    num_feat = size(population, 2) - NUM_EXTRA_COL
    insertcols!(population, :pred => 0.)
    insertcols!(population, :orig_ref => 1:nrow(population))
    # lazy update
    # template_feats = collect(Float64, orig_instance[1:num_feat])
    template_feats = copy(orig_instance)
    compressed_feats = Vector{Float64}(undef, size(population, 1) * num_feat)

    # we use the internal representation (possibly compressed, etc.) of the desired_class
    matrixified_entities::Array{Float64, 2} = MLJ.matrix(population[!,1:num_feat])
    changing_feats = Vector{Float64}(undef, num_feat)
    score = Vector{Float64}(undef, 32)
    for entities in grouped_entities
        changing_set::BitVector = entities[1,:mod]
        orig_refs::Vector{Int} = entities[!,:orig_ref]
        # template_feats = collect(Float64, entities[1,1:num_feat])
        for i=1:num_feat; if changing_set[i]; template_feats[i] = matrixified_entities[orig_refs[1],i]; end; end
        real_to_compressed_feats_map = Vector{Int}(undef, num_feat)
        compressed_feat_count = 0
        for i = 1:num_feat
            if changing_set[i]
                compressed_feat_count += 1
                real_to_compressed_feats_map[i] = compressed_feat_count
            end
        end
        begin
            i = 1
            num_entity = size(entities, 1)
            while i + 31 <= num_entity
                for j = 1:num_feat
                    if changing_set[j]
                        for k = 1:32
                            compressed_feats[(real_to_compressed_feats_map[j] - 1) * 32 + k] = matrixified_entities[orig_refs[i + k - 1], j]
                        end
                    end
                end
                res::Vector{Float64} = RapidScorer.eval_ensemble(
                    score,
                    rsensemble, compressed_feats, real_to_compressed_feats_map,changing_set,
                     template_feats, desired_class = desired_class_value)
                for j = 1:32
                    entities[i + j - 1, :pred] = res[j]
                    # pred[i + j - 1] = res[j]
                    # println("RS: ", res[j], " ", pdf(MLJ.predict(classifier, DataFrame(entities[i+j-1,1:end-3]))[1], desired_class))
                end
                i += 32
            end
            if i <= num_entity
                qs_changing_set = [i for (i, b) in enumerate(changing_set) if b]
                partialqsensemble::QuickScorer.PartialQSEnsemble{Float64,UInt32} = if haskey(cache, changing_set)
                    cache[changing_set]
                else
                    cache[changing_set] = QuickScorer.PartialQSEnsemble(qsensemble, template_feats, qs_changing_set, desired_class = desired_class_value)
                end
                # push!(cnt, count(changing_set))
                # partialqsensemble = QuickScorer.PartialQSEnsemble(qsensemble, template_feats, qs_changing_set, desired_class = desired_class_value)
                while i <= num_entity
                    for j = 1:num_feat
                        if changing_set[j]
                            changing_feats[real_to_compressed_feats_map[j]] = matrixified_entities[orig_refs[i], j]
                        end
                    end
                    # for j = 1:num_feat
                    #     changing_feats[j] = matrixified_entities[orig_refs[i], j]
                    # end
                    entities[i,:pred] = QuickScorer.eval_partial_ensemble(partialqsensemble, changing_feats)
                    # entities[i,:pred] = QuickScorer.eval_ensemble(qsensemble, changing_feats)
                    # pred[i] = QuickScorer.eval_partial_ensemble(partialqsensemble, changing_feats)
                    # println("QS: ", entities[i,:pred] - pdf(MLJ.predict(classifier, DataFrame(entities[i,1:end-3]))[1], desired_class))
                    i += 1
                end
            end
            for i=1:num_feat; if changing_set[i]; template_feats[i] = orig_instance[i]; end; end
        end
    end
    pred::Vector{Float64} = population[!, :pred]
    select!(population, Not([:pred, :orig_ref]))
    return pred
end

# column style, full entity, full q/rs
@inline function predict(ensemble::RandomForestEval, df::DataFrame)

    pred::Vector{Float64} = QuickScorer.eval_ensemble(ensemble.qsensemble, df)

    return pred
end

# row style, full entity, regular q/rs
@inline function predict(ensemble::RandomForestEval, df::DataFrame, _dummy) # to make it currently unused
    rsensemble = ensemble.rsensemble
    qsensemble = ensemble.qsensemble
    orig_instance = ensemble.orig_instance
    desired_class_value = ensemble.desired_value

    # Group the entities by their mod     # TODO: why are we grouping the entities?
    # grouped_entities = groupby(df, :mod)

    # Number of entities and features in the DF
    num_feat = size(df, 2) - NUM_EXTRA_COL
    num_entity = size(df, 1)

    # Matrix of entites
    entities::Array{Float64,2} = MLJ.matrix(df[!,1:end-3])

    # Lazy update
    # template_feats = collect(Float64, orig_instance[1:num_feat])
    template_feats = copy(orig_instance)
    compressed_feats = Vector{Float64}(undef, size(df, 1) * num_feat)
    real_to_compressed_feats_map = [i for i = 1:num_feat]

    # We use regular RS / QS  so all features are changing
    changing_set = trues(num_feat)
    changing_feats = Vector{Float64}(undef, num_feat)

    pred = Vector{Float64}(undef, num_entity)
    # print("->$(typeof(compressed_feats)), $(typeof(entities))<-")
    # println("->$(size(compressed_feats)), $(size(entities))<-")

    begin
        i = 1
        while i + 31 <= num_entity
            for j = 1:num_feat
                for k = 1:32
                    compressed_feats[(j - 1) * 32 + k] = entities[i + k - 1, j]
                end
            end
            res = RapidScorer.eval_ensemble(
                rsensemble, compressed_feats, real_to_compressed_feats_map,changing_set,
                template_feats, desired_class = desired_class_value)
            for j = 1:32
                pred[i + j - 1] = res[j]
            end
            i += 32
        end
        while i <= num_entity
            for j = 1:num_feat
                changing_feats[j] = entities[i, j]
            end
            pred[i] = QuickScorer.eval_ensemble(qsensemble, changing_feats, desired_class = desired_class_value)
            i += 1
        end
    end
    return pred
end

end
