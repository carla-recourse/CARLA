using Pkg; Pkg.activate("./carla/recourse_methods/catalog/geco/library/GeCo.jl") # change to the pkg with geco
using GeCo
import Pandas

using Base.Threads
using DataFrames

function pd_to_df(df_pd)
    colnames = map(Symbol, df_pd.columns)
    df = DataFrames.DataFrame(Any[Array(df_pd[c].values) for c in colnames], colnames)
    return df
end

function getPLAF(orig_instance, immutables)
    plaf = initPLAF()
    for immutable in immutables
        c = Float64(orig_instance[immutable])
        # println(c)
        plaf = @PLAF(plaf, :cf.aa .== parse(Float64, "$c"))

        cur_index = length(plaf.constraints)
        push!(plaf.constraints[cur_index].features, Symbol(immutable))
        deleteat!(plaf.constraints[cur_index].features, 1)

    end

    return plaf
end

function get_explanations(orig_instances_pd, X, classifier, immutables, desired_class, max_num_generations, min_num_generations, max_num_samples, norm_ratio)
    orig_instances_df = pd_to_df(orig_instances_pd)

    lengths = nrow(orig_instances_df)
    num_feature = ncol(orig_instances_df)
    if Threads.nthreads() > 1
        # we will do multi threads
        print("using multi core")
        explanations = Array{Union{Nothing, DataFrame}}(nothing, lengths)
        Threads.@threads for row_idx in 1:lengths
            orig_instance = orig_instances_df[row_idx,:]

            # build the plaf connstrains
            plaf = getPLAF(orig_instance, immutables)

            explanation, = explain(orig_instance, X, plaf, classifier, desired_class=desired_class, max_num_generations=max_num_generations, min_num_generations=min_num_generations, max_num_samples=max_num_samples, norm_ratio=norm_ratio)
            explanations[row_idx] = explanation
        end

        for i in 1:length
            push!(orig_instances_df, explanations[i][1,1:num_feature])
        end
    else
        for row_idx in 1:lengths
            orig_instance = orig_instances_df[row_idx,:]

            # build the plaf connstrains
            plaf = getPLAF(orig_instance, immutables)

            explanation, = explain(orig_instance, X, plaf, classifier, desired_class=desired_class, max_num_generations=max_num_generations, min_num_generations=min_num_generations, max_num_samples=max_num_samples, norm_ratio=norm_ratio)
            push!(orig_instances_df, explanation[1,1:num_feature])
        end
    end
    return Pandas.DataFrame(orig_instances_df[lengths+1:lengths*2, :])
end
