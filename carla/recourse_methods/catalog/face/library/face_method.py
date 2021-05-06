# import helpers
import numpy as np

# import Dijkstra's shortest path algorithm
from scipy.sparse import csgraph, csr_matrix

# import graph building methods
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph


def graph_search(
    data,
    index,
    keys_mutable,
    keys_immutable,
    continuous_cols,
    binary_cols,
    model,
    n_neighbors=50,
    p_norm=2,
    mode="knn",
    frac=0.4,
    radius=0.25,
):
    # This is our own implementation of these methods since the authors did not provide implementations online,
    # neither did they respond to our emails
    # We implemented the eps radius graph and the knn graph methods. The density graph will be added in the future.

    # This one implements the FACE method from
    # Rafael Poyiadzi et al (2020), "FACE: Feasible and Actionable Counterfactual Explanations",
    # Conference on AI, Ethics & Accountability (AIES, 2020)

    """
    :param data: df
    :param n_neighbors: int > 0; number of neighbors when constructing knn graph
    :param step: float > 0; step_size for growing spheres
    :param mode: str; either 'knn' or 'epsilon'
    :param model: classification model (either tf keras, pytorch or sklearn)
    :param p_norm: float=>1; denotes the norm (classical: 1 or 2)
    :param frac: float 0 < number =< 1; fraction of data for which we compute the graph; if frac = 1, and data set large, then compute long
    :param keys_mutable: list; list of input names we can search over
    :param keys_immutable: list; list of input names that may not be searched over
    :param radius: float > 0; parameter for epsilon density graph
    :return: candidate_counterfactual_star: np array (min. cost counterfactual explanation)
    """

    # Divide data in 'mutable' and 'non-mutable'
    # In particular, divide data in 'mutable & binary' and 'mutable and continuous'

    # Choose a subset of data for computational efficiency
    number_samples = np.int(np.rint(frac * data.values.shape[0]))
    chosen_indeces = np.random.choice(
        np.arange(0, index).tolist()
        + np.arange(index + 1, data.values.shape[0]).tolist(),
        replace=False,
        size=number_samples,
    )
    chosen_indeces = [
        index
    ] + chosen_indeces.tolist()  # make sure sample under consideration included
    data = data.iloc[chosen_indeces]
    data = data.sort_index()

    # ADD CONSTRAINTS by immutable inputs into adjacency matrix
    # if element in adjacency matrix 0, then it cannot be reached
    # this ensures that paths only take same sex / same race / ... etc. routes
    # TODO: CANNOT deal with continuous constraints (e.g. age) yet
    # TODO: However, authors also do not mention how to go about it
    for i in range(len(keys_immutable)):
        epsilon = 0.5  # avoids division by 0
        immutable_constraint_matrix = np.outer(
            data[keys_immutable[i]].values + epsilon,
            data[keys_immutable[i]].values + epsilon,
        )

        immutable_constraint_matrix1 = immutable_constraint_matrix / (
            (1 + epsilon) ** 2
        )
        immutable_constraint_matrix1 = ((immutable_constraint_matrix1 == 1) * 1).astype(
            float
        )

        immutable_constraint_matrix2 = immutable_constraint_matrix / (epsilon ** 2)
        immutable_constraint_matrix2 = ((immutable_constraint_matrix2 == 1) * 1).astype(
            float
        )

    # POSITIVE PREDICTIONS
    y_predicted = model.predict_proba(data.values)
    y_predicted = np.argmax(y_predicted, axis=1)
    y_positive_indeces = np.where(y_predicted == 1)

    # TODO: replace counterfactual search by function, which takes 'mode' as an argument
    if mode == "knn":

        # obtain candidate targets (CT); two conditions need to be met:
        # (1) CT needs to be predicted positively & (2) CT needs to have certain "density"
        # for knn I interpret 'certain density' as sufficient number of neighbours

        neihboorhood_size = 3  # chosen in ad-hoc fashion
        n_neighbors_list = [
            n_neighbors - neihboorhood_size,
            n_neighbors,
            n_neighbors + neihboorhood_size,
        ]
        candidate_counterfactuals_star = []

        for n in n_neighbors_list:

            # STEP 1 -- BUILD NETWORK GRAPH
            graph = kneighbors_graph(data.values, n_neighbors=n, n_jobs=-1)
            adjacency_matrix = graph.toarray()
            adjacency_matrix = np.multiply(
                adjacency_matrix,
                immutable_constraint_matrix1,
                immutable_constraint_matrix2,
            )  # element wise multiplication
            graph = csr_matrix(adjacency_matrix)

            # STEP 2 -- APPLY SHORTEST PATH ALGORITHM  ## indeces=index (corresponds to x^F)
            distances = csgraph.dijkstra(
                csgraph=graph, directed=False, indices=index, return_predecessors=False
            )
            distances[index] = np.inf  # avoid min. distance to be x^F itself
            min_distance = distances.min()

            # STEP 3 -- FIND COUNTERFACTUALS
            # minimum distance candidate counterfactuals
            candidate_min_distances = [
                min_distance,
                min_distance + 1,
                min_distance + 2,
                min_distance + 3,
            ]
            min_distance_indeces = np.array([0])
            for min_dist in candidate_min_distances:
                min_distance_indeces = np.c_[
                    min_distance_indeces, np.array(np.where(distances == min_dist))
                ]
            min_distance_indeces = np.delete(min_distance_indeces, 0)

            indeces_counterfactuals = np.intersect1d(
                np.array(y_positive_indeces), np.array(min_distance_indeces)
            )

            for i in range(indeces_counterfactuals.shape[0]):
                candidate_counterfactuals_star.append(
                    data.values[indeces_counterfactuals[i]]
                )

    elif mode == "epsilon":

        radius_neighboorhood_size = 0.10  # chosen in ad-hoc fashion
        radius_neighbors_list = [
            radius - radius_neighboorhood_size,
            radius,
            radius + radius_neighboorhood_size,
        ]
        candidate_counterfactuals_star = []

        for r in radius_neighbors_list:

            # STEP 1 -- BUILD NETWORK GRAPH
            graph = radius_neighbors_graph(data.values, radius=r, n_jobs=-1)
            adjacency_matrix = graph.toarray()
            adjacency_matrix = np.multiply(
                adjacency_matrix,
                immutable_constraint_matrix1,
                immutable_constraint_matrix2,
            )  # element wise multiplication
            graph = csr_matrix(adjacency_matrix)

            # STEP 2 -- APPLY SHORTEST PATH ALGORITHM  ## indeces=index (corresponds to x^F)
            distances = csgraph.dijkstra(
                csgraph=graph, directed=False, indices=index, return_predecessors=False
            )
            distances[index] = np.inf  # avoid min. distance to be x^F itself
            min_distance = distances.min()

            # STEP 3 -- FIND COUNTERFACTUALS
            # minimum distance candidate counterfactuals
            candidate_min_distances = [
                min_distance,
                min_distance + 1,
                min_distance + 2,
                min_distance + 3,
            ]
            min_distance_indeces = np.array([0])

            for min_dist in candidate_min_distances:
                min_distance_indeces = np.c_[
                    min_distance_indeces, np.array(np.where(distances == min_dist))
                ]
            min_distance_indeces = np.delete(min_distance_indeces, 0)

            indeces_counterfactuals = np.intersect1d(
                np.array(y_positive_indeces), np.array(min_distance_indeces)
            )

            for i in range(indeces_counterfactuals.shape[0]):
                candidate_counterfactuals_star.append(
                    data.values[indeces_counterfactuals[i]]
                )

    candidate_counterfactual_star = np.array(candidate_counterfactuals_star)

    # STEP 4 -- COMPUTE DISTANCES between x^F and candidate x^CF; else return NaN
    if candidate_counterfactual_star.size == 0:
        candidate_counterfactual_star = np.empty(
            data.values.shape[1],
        )
        candidate_counterfactual_star[:] = np.nan

    else:
        if p_norm == 1:
            c_dist = np.abs((data.values[index] - candidate_counterfactual_star)).sum(
                axis=1
            )
        elif p_norm == 2:
            c_dist = np.square(
                (data.values[index] - candidate_counterfactuals_star)
            ).sum(axis=1)
        else:
            print("Distance not defined yet")

        min_index = np.argmin(c_dist)
        candidate_counterfactual_star = candidate_counterfactual_star[min_index]

    return candidate_counterfactual_star
