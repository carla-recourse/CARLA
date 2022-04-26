# Implementation of the distance measure for the genetic algorithm

@inline absolute(val, orig_val, range)::Float64 = abs(val - orig_val) / range

function distance(df::DataFrame, orig_instance::DataFrameRow, num_features::Int64, featureRanges::Dict{Symbol, Float64}; #, norm_ratio::Array{Float64,1}, distance_temp::Array{Float64,1})
    norm_ratio::Array{Float64,1}=default_norm_ratio,
    distance_temp::Array{Float64,1}=zeros(Float64, 4*nrow(df)))::Array{Float64,1}

    if length(distance_temp) < size(df,1)*4
        @warn "We increase the size of distance_temp in selection operator"
        resize!(distance_temp, size(df,1)*4)
    end

    for i = 1:(4*size(df,1))
        @inbounds distance_temp[i] = 0.0
    end

    # compute the normalized absolute distance for each feature
    for (feature,col) in pairs(eachcol(df))

        # ad hoc hack
        feature in (:generation, :score, :outc, :mod, :estcf, :count, :distance) && continue


        if elscitype(col) != Multiclass     # Feature is not categorical

            data::Vector{Float64} = col
            orig_val::Float64 = orig_instance[feature]
            range = featureRanges[feature]

            # println(feature, elscitype(col), range, norm_ratio)

            row_index = 0
            for val in data
                # if numerical, get the absolute difference and then divided by the range of the feature
                diff::Float64 = abs(val - orig_val) / range ## absolute(val, orig_val, range)

                distance_temp[row_index + 1] += (diff != 0.0)                 ## zero norm
                distance_temp[row_index + 2] += diff                          ## one norm
                distance_temp[row_index + 3] += diff * diff                   ## two norm
                distance_temp[row_index + 4] = max(distance_temp[row_index + 4], diff)   ## inf norm

                row_index += 4
            end
        else
            orig_categ_val::Int64 = orig_instance[feature]

            row_index = 0
            for val in col
                # for the categorical -- 1 for they are not same and 0 for same
                diff::Bool = (val != orig_categ_val)

                distance_temp[row_index + 1] += diff                          ## zero norm
                distance_temp[row_index + 2] += diff                          ## one norm
                distance_temp[row_index + 3] += diff                          ## two norm
                distance_temp[row_index + 4] = max(distance_temp[row_index + 4], diff)   ## inf norm

                row_index += 4
            end
        end
    end

    return Float64[
        norm_ratio[1] * (distance_temp[(row * 4) + 1] / num_features) +
        norm_ratio[2] * (distance_temp[(row * 4) + 2] / num_features) +
        norm_ratio[3] * (sqrt(distance_temp[(row * 4) + 3] / num_features))  +
        norm_ratio[4] * distance_temp[(row * 4) + 4]
        for row in 0:(nrow(df)-1)
    ]
end

function distance(row::DataFrameRow, orig_instance::DataFrameRow, num_features::Int64, featureRanges::Dict{Symbol, Float64};
    norm_ratio::Array{Float64,1}=default_norm_ratio)::Float64

    dist = zeros(Float64, 4)
    for feature in keys(row)

        feature in (:generation, :score, :outc, :mod, :estcf, :count) && continue

        val = row[feature]
        if eltype(val) != Multiclass
            if haskey(featureRanges, feature)
                range = featureRanges[feature]
            else
                range = maximum(col) - minimum(col)
                featureRanges[feature] = range
            end

            diff = abs(orig_instance[feature] - val) / range   ## TODO: Check for type consistentcy here

            dist[1] += (diff != 0.0)       ## zero norm
            dist[2] += diff                ## one norm
            dist[3] += diff * diff         ## two norm
            dist[4] = max(dist[4], diff)   ## inf norm
        else
            diff = (orig_instance[feature] != val)

            dist[1] += diff                ## zero norm
            dist[2] += diff                ## one norm
            dist[3] += diff                ## two norm
            dist[4] = max(dist[4], diff)   ## inf norm
        end
    end

    return norm_ratio[1] * (dist[1] / num_features) +
        norm_ratio[2] * (dist[2] / num_features) +
        norm_ratio[3] * sqrt(dist[3] / num_features)  +
        norm_ratio[4] * dist[4]
end



## TODO: Move this somewhere else?
function minimumObservableCounterfactual(data, predictions, orig_instance, program;
    check_feasibility::Bool=false,
    desired_class::Int64=1,
    norm_ratio::Array{Float64,1}=[0.0,1.0,0.0,0.0],
    num_features::Int64=(ncol(data)-1),
    ranges::Dict{Symbol,Float64}=Dict(feature => Float64(maximum(col)-minimum(col)) for (feature, col) in pairs(eachcol(data))),
    distance_temp=Array{Float64,1}(undef, nrow(data)*4))

    selected_data = observableCounterfactuals(data, predictions, orig_instance, program.constraints, program.implications;
        check_feasibility=check_feasibility,
        desired_class=desired_class)

    if isempty(selected_data)
        return nothing, nothing
    end

    if length(distance_temp) < nrow(selected_data) * 4
        resize!(distance_temp, nrow(selected_data) * 4)
    end

    distances = distance(selected_data, orig_instance, num_features, ranges;
        norm_ratio=norm_ratio, distance_temp=distance_temp)

    row = argmin(distances)
    return selected_data[row, :], distances[row]
end

function observableCounterfactuals(data::DataFrame, predictions, orig_instance::DataFrameRow, constraints::Vector{Constraint}, implications::Vector{Implication};
    check_feasibility::Bool=false,
    desired_class::Int64=1)

    selected_data = data[predictions .== desired_class, :]

    if check_feasibility
        for constraint in constraints

            constraint_pair = constraint.features => constraint.fun(orig_instance)
            filter!(constraint_pair, selected_data)
        end

        for implication in implications
            filter!(!(implication.condition(orig_instance)), selected_data)
        end
    end

    return selected_data
end