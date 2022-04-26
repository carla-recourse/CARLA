## Generates the initial population
function initialPopulation(orig_instance, feasible_space; compress_data::Bool=false, max_num_samples::Int64=20)

    groups = feasible_space.groups
    num_features = feasible_space.num_features

    num_rows = length(groups) * max_num_samples

    if compress_data
        # Population is represented by the Δ-representation
        initial_pop = initializeManager(orig_instance; extended=true)
    else
        # Population is represented by a dataframe
        initial_pop = DataFrame(orig_instance)

        repeat!(initial_pop, num_rows)
        insertcols!(initial_pop,
            :score=>zeros(Float64, num_rows),
            :outc=>falses(num_rows),
            :estcf=>falses(num_rows),
            :mod=>BitVector[falses(num_features) for _=1:num_rows]
            )

        valid_rows = falses(num_rows)
        rownum = 0
    end

    for (index, group) in enumerate(groups)
        df = feasible_space.feasibleSpace[index]

        # if the feasible space is empty, continue to next group
        isempty(df) && continue;

        num_samples = min(max_num_samples, nrow(df))
        sampled_rows = StatsBase.sample(1:nrow(df), StatsBase.FrequencyWeights(df.count), num_samples; replace=false, ordered=true)

        if compress_data
            for s in 1:num_samples
                row = sampled_rows[s]
                push!(initial_pop, group.indexes, (df[row, 1:end-NUM_EXTRA_FEASIBLE_SPACE_COL]...,score=0.0, outc=false, estcf=false))
            end
        else
            for feature in group.features
                for s in 1:num_samples
                    initial_pop[rownum+s, feature] = df[sampled_rows[s], feature]
                end
            end
            for s in 1:num_samples
                initial_pop[rownum+s, :mod] .|= group.indexes

                valid_action = actionCascade(initial_pop[rownum+s,:], feasible_space.implications)
                valid_rows[rownum+s] = valid_action
            end
            rownum += num_samples
        end
    end

    if compress_data
        return initial_pop
    else
        return initial_pop[valid_rows, :]
    end
end