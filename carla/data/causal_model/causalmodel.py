import networkx as nx
from causalgraphicalmodels import StructuralCausalModel

from .synthethic_data import ScmDataset


def get_noise_string(node):
    if not node[0] == "x":
        raise ValueError
    return "u" + node[1:]


class CausalModel(object):
    def __init__(
        self,
        scm_class,
        structural_equations_np,
        structural_equations_ts,
        noise_distributions,
    ):
        self.scm_class = scm_class
        self.structural_equations_np = structural_equations_np
        self.structural_equations_ts = structural_equations_ts
        self.noise_distributions = noise_distributions

        self._scm = StructuralCausalModel(
            structural_equations_np
        )  # may be redundant, can simply call CausalGraphicalModel...
        self._cgm = self._scm.cgm

    def get_topological_ordering(self, node_type="endogenous"):
        tmp = nx.topological_sort(self._cgm.dag)
        if node_type == "endogenous":
            return tmp
        elif node_type == "exogenous":
            return ["u" + node[1:] for node in tmp]
        else:
            raise Exception(f"{node_type} not recognized.")

    def get_children(self, node):
        return set(self._cgm.dag.successors(node))

    def get_descendents(self, node):
        return nx.descendants(self._cgm.dag, node)

    def get_parents(self, node, return_sorted=True):
        tmp = set(self._cgm.dag.predecessors(node))
        return sorted(tmp) if return_sorted else tmp

    def get_ancestors(self, node):
        return nx.ancestors(self._cgm.dag, node)

    def get_non_descendents(self, node):
        return (
            set(nx.topological_sort(self._cgm.dag))
            .difference(self.get_descendents(node))
            .symmetric_difference(set([node]))
        )

    def generate_dataset(self, size):
        return ScmDataset(self, size)

    def visualize_graph(self, experiment_folder_name=None):
        if experiment_folder_name:
            save_path = f"{experiment_folder_name}/_causal_graph"
            view_flag = False
        else:
            save_path = "_tmp/_causal_graph"
            view_flag = True
        self._cgm.draw().render(save_path, view=view_flag)

    def print(self):
        raise NotImplementedError
