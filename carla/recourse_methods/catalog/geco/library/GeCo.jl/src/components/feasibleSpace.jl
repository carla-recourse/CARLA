using MLJScientificTypes

struct FeatureGroup
    features::Tuple{Vararg{Symbol}}
    names::Vector{Symbol}
    indexes::BitVector
    allCategorical::Bool
end

## Grounded Implications includes:
# Feasible space
# Condition
# Consequence
# Features in Condition / Consequence

# Assumptions:
# - consequence is only one comparison not several
# - consequence is contained in one feature group
# - constraints are always contained in one FeatureGroup

struct GroundedImplication
    condition::Function
    sampleSpace::DataFrame
    condFeatures::BitVector
    conseqFeatures::Vector{Symbol}
    conseqFeaturesBitVec::BitVector
end

struct FeasibleSpace
    groups::Vector{FeatureGroup}
    ranges::Dict{Symbol,Float64}
    num_features::Int64
    feasibleSpace::Vector{DataFrame}
    implications::Vector{GroundedImplication}
end


function initGroups(prog::PLAFProgram, data::DataFrame)

    #feature_list = Array{Feature,1}()
    groups = Array{FeatureGroup, 1}()

    addedFeatures = Set{Symbol}()
    onehotFeature = falses(ncol(data))

    for group in prog.groups

        allCategorical = true

        indexes = falses(ncol(data))

        for (idx, feature) in enumerate(group)
            @assert feature ∈ propertynames(data) "The feature $feature does not occur in the input data."
            @assert feature ∉ addedFeatures "Each feature can be in at most one group!"
            push!(addedFeatures, feature)
            indexes[findfirst(isequal(feature),propertynames(data))] = true
            allCategorical = (elscitype(data[!,feature]) == Multiclass)
        end

        push!(groups,
            FeatureGroup(
                group,
                [group...],
                indexes,
                allCategorical
                )
            )
    end

    for feature in propertynames(data)
        if feature ∉ addedFeatures
            @assert feature ∈ propertynames(data) "The feature $feature does not occur in the input data."

            indexes = falses(ncol(data))
            indexes[findfirst(isequal(feature),propertynames(data))] = true

            push!(groups,
                FeatureGroup(
                    (feature,),
                    [feature],
                    indexes,
                    elscitype(data[!,feature]) == Multiclass
                    )
            )
        end
    end

    return groups
end

function initDomains(prog::PLAFProgram, data::DataFrame)::Vector{DataFrame}
    groups = initGroups(prog, data)
    return initDomains(groups, data)
end

function initDomains(groups::Vector{FeatureGroup}, data::DataFrame)::Vector{DataFrame}
    domains = Vector{DataFrame}(undef, length(groups))
    for (gidx, group) in enumerate(groups)
        feat_syms = group.names
        if length(feat_syms) > 1000
            domains[gidx] = data[:, feat_syms]
            domains[gidx].count = ones(Int64, nrow(data))
            domains[gidx] = unique(domains[gidx])
        else
            domains[gidx] = combine(groupby(data[!,feat_syms], feat_syms), nrow => :count)
        end
    end
    return domains
end

function groundImplications(
    implications::Vector{Implication},
    groups::Vector{FeatureGroup},
    feasible_space::Vector{DataFrame},
    orig_instance::DataFrameRow)::Vector{GroundedImplication}

    conseqGroupDependency = zeros(Int64, length(groups))

    condGroupDependency = [Int64[] for _ in implications]
    condGroupDependencyBitVec = [falses(length(groups)) for _ in implications]

    dependson = falses(length(groups), length(groups))

    for (impID,implication) in enumerate(implications)
        # Find feature group that contains the features in the consequence of the implication
        conseqGroupDependency[impID] = findfirst(g -> (implication.conseqFeatures ⊆ g.features), groups)

        # Find all feature groups that contains the features in the condition of the implication
        condGroupDependency[impID] = findall(g -> !isempty(implication.condFeatures ∩ g.features), groups)

        for gid in condGroupDependency[impID]
            condGroupDependencyBitVec[impID][gid] = true
            dependson[conseqGroupDependency[impID], gid] = true
        end
    end

    # println("dependson matrix: ")
    # println(dependson)

    S = Set{Int64}(gid for gid in 1:length(groups) if all(.!dependson[gid, :]))

    processedGroups = falses(length(groups))
    processedImplication = falses(length(implications))

    groundedImplications = Vector{GroundedImplication}()

    while !isempty(S)
        gid = pop!(S)

        processedGroups[gid] = true

        for (impID, implication) in enumerate(implications)

            if !processedImplication[impID] &&
                ((processedGroups .& condGroupDependencyBitVec[impID]) == condGroupDependencyBitVec[impID])

                condFeatures = falses(length(groups[1].indexes))
                for gid in condGroupDependency[impID]
                    condFeatures .|= groups[gid].indexes
                end

                conseq_gid = conseqGroupDependency[impID]

                # Filter the corresponding feasible space to define the sample space
                conseq_func = implication.conseqFeatures => implication.consequence(orig_instance)
                fspace = filter(conseq_func, feasible_space[conseq_gid])

                push!(groundedImplications,
                    GroundedImplication(
                        implication.condition(orig_instance),
                        fspace,
                        condFeatures,
                        groups[conseq_gid].names,
                        groups[conseq_gid].indexes)
                )

                processedImplication[impID] = true
            end
        end

        for i in 1:length(groups)
            if !processedGroups[i] && all(processedGroups[dependson[i,:]])
                push!(S, i)
            end
        end
    end

    @assert all(processedImplication) "Some implications have not been processed, most likely due to a loop in the dependencies of the implipactions."

    return groundedImplications
end

function feasibleSpace(data::DataFrame, orig_instance::DataFrameRow, prog::PLAFProgram;
    domains::Vector{DataFrame}=Vector{DataFrame}())::FeasibleSpace

    constraints = prog.constraints

    groups = initGroups(prog, data)
    ranges = Dict(feature => max(1.0, Float64(maximum(col)-minimum(col))) for (feature, col) in pairs(eachcol(data)))
    num_features = ncol(data)

    feasible_space = Vector{DataFrame}(undef, length(groups))

    if isempty(domains)
        domains = initDomains(groups, data)
    else
        @assert length(domains) == length(groups)
    end

    satisfied_constraints = falses(length(constraints))

    for (gidx, group) in enumerate(groups)

        features = group.names
        fspace = filter(row -> row[features] != orig_instance[features], domains[gidx])

        for (cidx, constraint) in enumerate(constraints)
            if constraint.features ⊆ group.features
                @assert !satisfied_constraints[cidx]

                # Compute constraints on this group
                constraint_pair = constraint.features => constraint.fun(orig_instance)
                filter!(constraint_pair, fspace)
                satisfied_constraints[cidx] = true
            end
        end

        fspace.distance = distance(fspace, orig_instance, num_features, ranges)
        feasible_space[gidx] = fspace
    end

    for (cidx, constraint) in enumerate(constraints)
        if !satisfied_constraints[cidx]
            println("We have an extra constraint for columns: $(constraint[1])")
        end
    end

    groundedImplications = groundImplications(prog.implications, groups, feasible_space, orig_instance)

    return FeasibleSpace(groups, ranges, num_features, feasible_space, groundedImplications)
end




# function feasibleSpace2(orig_instance::DataFrameRow, groups::Vector{FeatureGroup}, domains::Vector{DataFrame})::Vector{DataFrame}
#     # We create a dictionary of constraints for each feature
#     # A constraint is a function
#     feasibility_constraints = Dict{String,Function}()
#     feasible_space = Vector{DataFrame}(undef, length(groups))

#     @assert length(domains) == length(groups)

#     # This should come from the constraint parser
#     # actionable_features = []

#     for (gidx, group) in enumerate(groups)

#         domain = domains[gidx]

#         feasible_rows = trues(nrow(domain))
#         identical_rows = trues(nrow(domain))

#         for feature in group.features

#             orig_val::Float64 = orig_instance[feature.name]
#             col::Vector{Float64} = domain[!, feature.name]

#             identical_rows .&= (col .== orig_val)

#             if !feature.actionable
#                 feasible_rows .&= (col .== orig_val)
#             elseif feature.constraint == INCREASING
#                 feasible_rows .&= (col .> orig_val)
#             elseif feature.constraint == DECREASING
#                 feasible_rows .&= (col .< orig_val)
#             end
#         end

#         # Remove the rows that have identical values - we don't want to sample the original instance
#         feasible_rows .&= .!identical_rows

#         feasible_space[gidx] = domains[gidx][feasible_rows, :]
#         feasible_space[gidx].distance = distanceFeatureGroup(feasible_space[gidx], orig_instance, group)
#     end

#     return feasible_space
# end

# function feasibleSpace2(orig_instance::DataFrameRow, groups::Vector{FeatureGroup}, data::DataFrame)::Vector{DataFrame}
#     # We create a dictionary of constraints for each feature
#     # A constraint is a function
#     feasibility_constraints = Dict{String,Function}()
#     feasible_space = Vector{DataFrame}(undef, length(groups))

#     # This should come from the constraint parser
#     actionable_features = []

#     feasible_rows = trues(size(data,1))
#     identical_rows = trues(size(data,1))

#     for (gidx, group) in enumerate(groups)

#         fill!(feasible_rows, true)
#         fill!(identical_rows, true)

#         feat_names = String[]
#         for feature in group.features
#             identical_rows .&= (data[:, feature.name] .== orig_instance[feature.name])

#             if !feature.actionable
#                 feasible_rows .&= (data[:, feature.name] .== orig_instance[feature.name])
#             elseif feature.constraint == INCREASING
#                 feasible_rows .&= (data[:, feature.name] .> orig_instance[feature.name])
#             elseif feature.constraint == DECREASING
#                 feasible_rows .&= (data[:, feature.name] .< orig_instance[feature.name])
#             end
#             push!(feat_names, feature.name)
#         end

#         # Remove the rows that have identical values - we don't want to sample the original instance
#         feasible_rows .&= .!identical_rows

#         feasible_space[gidx] = combine(
#                 groupby(
#                     data[feasible_rows, feat_names],
#                     feat_names),
#                 nrow => :count
#                 )

#         feasible_space[gidx].distance = distanceFeatureGroup(feasible_space[gidx], orig_instance, group)
#     end

#     return feasible_space
# end
