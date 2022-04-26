
#######
### Crossover operator
#######
function crossover!(population::DataFrame, orig_instance::DataFrameRow, feasible_space::FeasibleSpace)

    feature_groups = feasible_space.groups
    sample_space = feasible_space.feasibleSpace
    num_features = feasible_space.num_features
    ranges = feasible_space.ranges

    gb = groupby( population[population.outc .== true,:], :mod )

    num_groups = size(keys(gb),1)

    for group1 in 1:num_groups
        parent1 = gb[group1][1,:]

        selective_mutation = repeat(DataFrame(parent1), length(feature_groups)+1)
        selective_mutation[:,:estcf] .= false

        valid_mutations = falses(length(feature_groups)+1)

        for (index, group) in enumerate(feature_groups)
            df = sample_space[index]

            # check whether we can change the feature value to something else
            (isempty(df) || group.allCategorical || !any(parent1.mod[group.indexes])) && continue

            space = df[df.distance .< distance(parent1,orig_instance,num_features,ranges), :]
            isempty(space) && continue

            sampled_row = StatsBase.sample(1:nrow(space), StatsBase.FrequencyWeights(space.count), 1)

            selective_mutation[index+1,group.names] = space[sampled_row,group.names]
            valid_mutations[index+1] = actionCascade(selective_mutation[index+1,:], feasible_space.implications)

            sampled_row = StatsBase.sample(1:nrow(space), StatsBase.FrequencyWeights(space.count), 1)
            selective_mutation[1,group.names] = space[sampled_row, group.names]  ## This might create duplicates!!

            valid_mutations[1] = actionCascade(selective_mutation[1,:], feasible_space.implications)
        end

        append!(population, selective_mutation[valid_mutations, :])

        for group2 in group1+1:num_groups

            parent2 = gb[group2][1,:]

            modified_features::BitVector = parent1.mod .| parent2.mod

            push!(population, parent1)
            crossover_candidate = population[end, :]
            crossover_candidate.estcf = false
            crossover_candidate.mod = parent1.mod .| parent2.mod

            for (index, group) in enumerate(feature_groups)
                df = sample_space[index]

                # check whether we can change the feature value to something else
                (isempty(df) || !any(modified_features[group.indexes])) && continue

                changed_p1 = any(parent1.mod[group.indexes])
                changed_p2 = any(parent2.mod[group.indexes])

                if changed_p1 && changed_p2
                    crossover_candidate[group.names] = (rand(Bool) ? parent1[group.names] : parent2[group.names])
                elseif changed_p1
                    crossover_candidate[group.names] = parent1[group.names]
                elseif changed_p2
                    crossover_candidate[group.names] = parent2[group.names]
                end
            end

            valid_action = actionCascade(crossover_candidate, feasible_space.implications)
            valid_action && push!(population, crossover_candidate)
        end
    end
end


function crossover!(manager::DataManager, orig_instance::DataFrameRow, feasible_space::FeasibleSpace)

    feature_groups = feasible_space.groups
    sample_space = feasible_space.feasibleSpace

    mod_list = collect(keys(manager.dict))
    num_groups = length(mod_list)

    placeholder=DataFrame(orig_instance)
    placeholder.score=0.0
    placeholder.outc=false
    placeholder.estcf=false

    for group1 in 1:num_groups

        mod_parent1 = mod_list[group1]

        population = manager.dict[mod_parent1]

        parent1 = population[1,:]
        # cols_parent1 = propertynames(parent1)[1:end-3]

        push!(population,parent1)
        population[end,:estcf] = false
        added_offspring = size(population,1)

        ## Selective Mutation:
        for (index, group) in enumerate(feature_groups)
            df = sample_space[index]
            # check whether we can change the feature value to something else
            (isempty(df) || group.allCategorical || !any(mod_parent1[group.indexes])) && continue

            group_dist = distance(parent1,orig_instance,feasible_space.num_features,feasible_space.ranges)
            rows::BitVector = df.distance .< group_dist

            space::DataFrame = df[rows, :]

            isempty(space) && continue

            sampled_row = StatsBase.sample(1:nrow(space), StatsBase.FrequencyWeights(space.count))
            push!(population,parent1)            # TODO: Do we want to do sample more cases here??
            population[end,group.names] = space[sampled_row, group.names]
            population[end,:estcf] = false

            sampled_row = StatsBase.sample(1:nrow(space), StatsBase.FrequencyWeights(space.count))
            population[added_offspring,group.names] = space[sampled_row, group.names]  ## This may add a duplicate!
        end

        ## Crossover:
        for group2 in group1+1:num_groups
            # println(group1, " ", group2)

            mod_parent2 = mod_list[group2]
            parent2 = manager.dict[mod_parent2][1,:]
            # cols_parent2 = propertynames(parent2)[1:end-3]

            modified_features::BitVector = mod_parent1 .| mod_parent2

            ## TODO: Can we improve on constructing the DF and pushing the placeholder?
            #push!(manager, modified_features, (orig_instance[modified_features]..., score=0.0, outc=false, estcf=false))
            push!(manager, modified_features, placeholder[1,:])

            d = get_store(manager, modified_features)
            crossover_candidate = d[end, :]

            for (index, group) in enumerate(feature_groups)
                df = sample_space[index]

                # check whether we can change the feature value to something else
                (isempty(df) || !any(modified_features[group.indexes])) && continue

                p1_changed = any(mod_parent1[group.indexes])
                p2_changed = any(mod_parent2[group.indexes])

                if p1_changed && p2_changed
                    if rand(Bool)
                        # cols = [n for n in group.names if n in cols_parent1]
                        crossover_candidate[group.names] = parent1[group.names]
                    else
                        # cols = [n for n in group.names if n in cols_parent2]
                        crossover_candidate[group.names] = parent2[group.names]
                    end
                elseif p1_changed
                    # cols = [n for n in group.names if n in cols_parent1]
                    crossover_candidate[group.names] = parent1[group.names]
                elseif p2_changed
                    # cols = [n for n in group.names if n in cols_parent2]
                    crossover_candidate[group.names] = parent2[group.names]
                end
            end
        end
    end

    actionCascade(manager, feasible_space.implications)
end
