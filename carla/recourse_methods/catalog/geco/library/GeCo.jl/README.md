# GeCo: Quality Counterfactual Explanations in Real Time

To run GeCo, you need to have Julia installed ([link](https://julialang.org/downloads/)). Then you can run the following commands to load the package.

```Julia
using Pkg; Pkg.activate(".")
using GeCo
```

We provide scripts to load the data and model. For instance:
```Julia
include("scripts/credit/credit_setup_MACE.jl");
include("scripts/adult/adult_setup_MACE.jl");
include("scripts/yelp/yelp_setup_PRF.jl");
include("scripts/allstate/allstate_setup_PRF.jl");
```

Then run the following command to compute the explanations:
```Julia
explanation,  = @time explain(orig_entity, X, path, classifier)
```
where `orig_entity` is the entity to be explained, `X` is the dataset, `path` is the path to the directory to the dataset, `classifier` is the model that is explained.

The explain function accepts the following optional keyword parameters:
* `desired_class` -- value of the desired output (default = 1)
* `k` (Int64) -- number of selected candidates during selection (default = 100)
* `max_num_generations` (Int64) -- maximum number of generations for genetic algorithm (default = 100)
* `min_num_generations` (Int64) -- minimum number of generations for genetic algorithm (default = 3)
* `max_num_samples` (Int64) -- maximum number of samples during mutation (default = 5)
* `convergence_k` (Int64) -- k used to check convergence (default = 5)
* `norm_ratio` -- parameters used for the distance function ([0.25, 0.25, 0.25, 0.25])
* `domains` -- precomputed active domain for each feature group (default = nothing)
* `compress_data` (Bool) -- use Delta representation (default = true)
* `return_df` (Bool) -- return a DataFrame instead of compressed Delta representation (default = false)
* `verbose` (Bool) -- verbose output (default = false)

To print out the proposed actions for the top counterfactuals run:
```Julia
actions(explanation, orig_instance)
```

If you would like to precompute the domains run:
```Julia
domains = initDomains(path, X)
```
