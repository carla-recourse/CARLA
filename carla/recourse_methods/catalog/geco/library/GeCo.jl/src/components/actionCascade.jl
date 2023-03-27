

function actionCascade(instance::DataFrameRow, implications::Vector{GroundedImplication}; dataManager::Bool=false)
    validAction = true
    for implication in implications
        if  implication.condition(instance)

            ## Assumption: There is only a single feature group that we need to change
            fspace = implication.sampleSpace

            isempty(fspace) && (validAction = false; break)

            sampled_row = StatsBase.sample(1:nrow(fspace), StatsBase.FrequencyWeights(fspace.count))
            features = implication.conseqFeatures

            instance[features] = fspace[sampled_row, features]
            instance[:mod] .|= implication.conseqFeaturesBitVec
        end
    end

    # !validAction &&  println("This action is not valid")
    return validAction
end


function actionCascade(manager::DataManager, implications::Vector{GroundedImplication})

    dict = manager.dict
    for impl in implications

        affect_bits = impl.condFeatures .| impl.conseqFeaturesBitVec

        for (mod, df) in dict

            # continue if dictionary is unrelated to the affected features
            !any(mod[affect_bits]) && continue

            # calculate validInstances
            validInstances = .!impl.condition.(DeltaDataFrameWrapper(instance, manager.orig_instance) for instance in eachrow(df))

            # store valid instances for next round
            if any(validInstances)
                dict[mod] = df[validInstances, :]
            end

            # there are invalid instances and we should process them and store for the next round
            if !all(validInstances) & !isempty(impl.sampleSpace)

                # println("There are $(count(.!validInstances)) invalid cases, below are one example")
                # println(df[.!validInstances, :][1, :])

                # tweak and add previously invalid instances
                refined_mod = copy(mod)
                refined_mod .|= impl.conseqFeaturesBitVec
                # note that refined_mod could be the same as mod, so we should update dict[mod] first
                # with validInstances before moving to invalid instances
                df_to_add = get_store_impl(dict, refined_mod, manager.orig_instance, manager.extended)

                fspace = impl.sampleSpace
                features = impl.conseqFeatures
                num_tuples = length(validInstances) - count(validInstances)
                sampled_rows = StatsBase.sample(1:nrow(fspace), StatsBase.FrequencyWeights(fspace.count), num_tuples)

                # during hcat, DataFrame will check that the columns from different dfs should be different, so
                # we need to exclude common columns first

                original_keys = keys(manager.orig_instance)[(mod .⊻ impl.conseqFeaturesBitVec) .& mod]
                manager.extended && push!(original_keys, :score, :outc, :estcf)

                append!(df_to_add, hcat(
                    df[.!validInstances, original_keys],
                    fspace[sampled_rows, features]))
            end
        end
    end
end
