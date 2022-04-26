
####
## The mutation operator
####

function mutation!(population::DataFrame, feasible_space::FeasibleSpace; max_num_samples::Int64 = 5)

    groups::Vector{FeatureGroup} = feasible_space.groups
    sample_space::Vector{DataFrame} = feasible_space.feasibleSpace

    estcfs = population.estcf::BitVector

    row = 1
    while row < nrow(population) && estcfs[row]
        entity = population[row,:]
        modified_features::BitVector = entity.mod::BitVector

        # The three lines below are to avoid deepcopies and pushing to DataFrames
        num_rows = length(groups) * max_num_samples
        mutatedInstances = DataFrame(entity)
        repeat!(mutatedInstances, num_rows)
        for i=1:num_rows
            mutatedInstances.mod[i] = BitArray{1}(modified_features)
            mutatedInstances.estcf[i] = false
        end

        # This BitVector is used to determine which mutations are valid
        validInstances = falses(num_rows)

        num_mutated_rows = 0
        for (index,group)  in enumerate(groups)
            df = sample_space[index]

            ## TODO: Test performance for modified_features[group.indexes] vs modified_features .& group.indexes
            (isempty(df) || any(modified_features[group.indexes])) && continue;

            num_samples = min(max_num_samples, nrow(df))

            weights::Vector{Int64} = df.count
            sampled_rows = StatsBase.sample(1:nrow(df), StatsBase.FrequencyWeights(weights), num_samples; replace=false, ordered=true)


            ## TODO: Test performance of for loop vs columnar approach
            for fname in group.names
                for s in 1:num_samples
                    mutatedInstances[num_mutated_rows+s, fname] = df[sampled_rows[s], fname]
                end
            end

            for s in 1:num_samples
                mutatedInstances[num_mutated_rows+s, :mod] .|= group.indexes

                valid_action = actionCascade(mutatedInstances[num_mutated_rows+s, :], feasible_space.implications)
                # !valid_action && println("We found an invalid action: ", valid_action)
                validInstances[num_mutated_rows+s] = valid_action
            end

            num_mutated_rows += num_samples
        end

        # @assert sum(validInstances) == num_mutated_rows
        # append!(population, mutatedInstances[1:num_mutated_rows, :])
        append!(population, mutatedInstances[validInstances, :])
        row += 1
    end
end

function mutation!(manager::DataManager, feasible_space::FeasibleSpace; max_num_samples::Int64 = 5)

    groups::Vector{FeatureGroup} = feasible_space.groups
    sample_space::Vector{DataFrame} = feasible_space.feasibleSpace

    keyset = collect(keys(manager))

    for mod in keyset

        population = manager.dict[mod]

        row = 1
        while row <= size(population, 1) && population[row, :estcf]
            entity_df = DataFrame(population[row, :])
            entity_df.estcf[1] = false
            repeat!(entity_df, max_num_samples)

            for (index,group)  in enumerate(groups)
                df = sample_space[index]

                (isempty(df) || any(mod[group.indexes])) && continue;

                num_samples = min(max_num_samples, nrow(df))

                weights::Vector{Int64} = df.count
                sampled_rows = StatsBase.sample(1:nrow(df), StatsBase.FrequencyWeights(weights), num_samples; replace=false, ordered=true)

                refined_modified_features = mod .| group.indexes

                # Would it be safe to use copycols=false here?
                mutatedInstances = hcat(
                    entity_df[1:num_samples,:],
                    df[sampled_rows, 1:end-NUM_EXTRA_FEASIBLE_SPACE_COL])
                # mutatedInstances[:,:mod] = (refined_modified_features for i=1:nrow(mutatedInstances))

                append!(manager, refined_modified_features, mutatedInstances)
            end

            row += 1
        end
    end

    actionCascade(manager, feasible_space.implications)
end
