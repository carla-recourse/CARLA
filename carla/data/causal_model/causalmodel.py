import networkx as nx
from causalgraphicalmodels import CausalGraphicalModel, StructuralCausalModel

from carla.data.load_scm import load_scm_equations

from .synthethic_data import ScmDataset


class CausalModel:
    """
    Class with topological methods given a structural causal model.

    Parameters
    ----------
    scm_class: str
        Name of the structural causal model

    Attributes
    ----------
    scm
    cgm
    scm_class
    structural_equations_np
    structural_equations_ts
    noise_distributions

    """

    def __init__(
        self,
        scm_class: str,
    ):
        self._scm_class = scm_class

        (
            self._structural_equations_np,
            self._structural_equations_ts,
            self._noise_distributions,
        ) = load_scm_equations(scm_class)

        self._scm = StructuralCausalModel(self._structural_equations_np)
        self._cgm = self._scm.cgm

    def get_topological_ordering(self, node_type="endogenous"):
        """Returns a generator of nodes in topologically sorted order.

        A topological sort is a nonunique permutation of the nodes such that an
        edge from u to v implies that u appears before v in the topological sort
        order.

        Parameters
        ----------
        node_type: str
            "endogenous" or "exogenous", i.e. nodes with "x" or "u" prefix respectively

        Returns
        -------
        iterable
            An iterable of node names in topological sorted order.
        """
        tmp = nx.topological_sort(self._cgm.dag)
        if node_type == "endogenous":
            return tmp
        elif node_type == "exogenous":
            return ["u" + node[1:] for node in tmp]
        else:
            raise Exception(f"{node_type} not recognized.")

    def get_children(self, node: str) -> set:
        """Returns an iterator over successor nodes of n.

        A successor of n is a node m such that there exists a directed
        edge from n to m.

        Parameters
        ----------
        node: str
            A node in the graph
        """
        return set(self._cgm.dag.successors(node))

    def get_parents(self, node: str, return_sorted: bool = True):
        """Returns an set over predecessor nodes of n.

        A predecessor of n is a node m such that there exists a directed
        edge from m to n.

        Parameters
        ----------
        node : str
           A node in the graph
        return_sorted : bool
            Return the set sorted
        """
        tmp = set(self._cgm.dag.predecessors(node))
        return sorted(tmp) if return_sorted else tmp

    def get_ancestors(self, node: str) -> set:
        """Returns all nodes having a path to `node`.

        Parameters
        ----------
        node : str
            A node in the graph

        Returns
        -------
        set()
            The ancestors of node
        """
        return nx.ancestors(self._cgm.dag, node)

    def get_descendents(self, node: str) -> set:
        """Returns all nodes reachable from `node`.

        Parameters
        ----------
        node : str
            A node in the graph

        Returns
        -------
        set()
            The descendants of `node`
        """
        return nx.descendants(self._cgm.dag, node)

    def get_non_descendents(self, node: str) -> set:
        """Returns all nodes not reachable from `node`.

        Parameters
        ----------
        node : str
            A node in the graph

        Returns
        -------
        set()
            The non-descendants of `node`
        """
        return (
            set(nx.topological_sort(self._cgm.dag))
            .difference(self.get_descendents(node))
            .symmetric_difference(set([node]))
        )

    def generate_dataset(self, size: int) -> ScmDataset:
        """Generates a Data object using the structural causal equations

        Parameters
        ----------
        size: int
            Number of samples in the dataset

        Returns
        -------
        ScmDataset
            a Data object filled with samples
        """
        return ScmDataset(self, size)

    def visualize_graph(self, experiment_folder_name=None):
        if experiment_folder_name:
            save_path = f"{experiment_folder_name}/_causal_graph"
            view_flag = False
        else:
            save_path = "_tmp/_causal_graph"
            view_flag = True
        self._cgm.draw().render(save_path, view=view_flag)

    @property
    def scm(self) -> StructuralCausalModel:
        return self._scm

    @property
    def cgm(self) -> CausalGraphicalModel:
        return self._cgm

    @property
    def scm_class(self) -> str:
        return self._scm_class

    @property
    def structural_equations_np(self) -> dict:
        return self._structural_equations_np

    @property
    def structural_equations_ts(self) -> dict:
        return self._structural_equations_ts

    @property
    def noise_distributions(self) -> dict:
        return self._noise_distributions
