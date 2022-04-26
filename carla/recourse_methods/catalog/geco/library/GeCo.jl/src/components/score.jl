# Implementations of the score function, which overloads prediction functions for different ML packages
import Pandas
function score(classifier::MLJ.Machine, counterfactuals::DataFrame, desired_class; extra_col = NUM_EXTRA_COL)::Vector{Float64}
    return broadcast(MLJ.pdf, MLJ.predict(classifier, counterfactuals[!, 1:end-extra_col]), desired_class)
end

function score(classifier::PyCall.PyObject, counterfactuals::DataFrame, desired_class; extra_col = NUM_EXTRA_COL)::Vector{Float64}
    if occursin("sklearn.neural_network._multilayer_perceptron", classifier.__module__) # uses `occursin` instead of `contains` because `contains` is only defined for julia 1.5 and above
        return MLPEvaluation.predict(classifier.coefs_, classifier.intercepts_, classifier.activation, MLJ.matrix(counterfactuals[!, 1:end-extra_col]))
    elseif occursin("sklearn", classifier.__module__)
        return ScikitLearn.predict_proba(classifier, MLJ.matrix(counterfactuals[!, 1:end-extra_col]))[:, desired_class+1]
    elseif occursin("torch", classifier.__module__)
        torch = pyimport("torch")
        in = torch.tensor(convert(Matrix, counterfactuals[!, 1:end-extra_col])).float()
        preds = classifier(in).float()
        return preds.detach().numpy()[:,desired_class+1]
    elseif occursin("carla", classifier.__module__)
        # add support for the carla benchmark
        preds = classifier.predict(Pandas.DataFrame(counterfactuals[!, 1:end-extra_col]))
        res = Array{Float64,1}()
        for pred in preds
            push!(res, pred[1])
        end
        return res
    end
    @error "We only support ScikitLearn, Torch and carla models for now. You can add more models by yourself if you want."
    return nothing
end

function score(classifier::PartialRandomForestEval, counterfactuals::DataFrame, desired_class)::Vector{Float64}
    return RandomForestEvaluation.predict(classifier, counterfactuals)
end

function score(classifier::PartialRandomForestEval, counterfactuals::DataManager, desired_class)::Vector{Float64}
    return RandomForestEvaluation.predict(classifier, counterfactuals)
end

function score(classifier::PartialMLPEval, counterfactuals::DataManager, desired_class)::Vector{Float64}
    return MLPEvaluation.predict(classifier, counterfactuals)
end

function score(classifier::PartialMLPEval, counterfactuals::DataFrame, desired_class)::Vector{Float64}
    return MLPEvaluation.predict(classifier, counterfactuals)
end

function score(classifier::RandomForestEval, counterfactuals::DataFrame, desired_class)::Vector{Float64}
    return RandomForestEvaluation.predict(classifier, counterfactuals)
end

function score(classifier::Function, counterfactuals::DataFrame, desired_class; extra_col = NUM_EXTRA_COL)::Vector{Float64}
    return classifier(counterfactuals)
end
