from sklearn.tree import _tree

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.catalog.parse_xgboost import (
    TREE_LEAF,
    TREE_UNDEFINED,
    _get_tree_from_booster,
    _parse_node,
    parse_booster,
    re_feature,
    re_numbers,
)


def test_constants():
    assert _tree.TREE_LEAF == TREE_LEAF
    assert _tree.TREE_UNDEFINED == TREE_UNDEFINED


def test_regex():
    # 'node_id:leaf=value'
    leaf_str = "3:leaf=0.5"

    leaf_numbers = [
        float(x) if "." in x else int(x) for x in re_numbers.findall(leaf_str)
    ]
    assert leaf_numbers == [3, 0.5]

    # 'node_id:[split] yes=left_child_id,no=right_child_id,missing=?'
    node_str = "0:[feature<0.6] yes=1,no=2,missing=-1"
    node_numbers = [
        float(x) if "." in x else int(x) for x in re_numbers.findall(node_str)
    ]
    feature = re_feature.findall(node_str)[0]

    assert node_numbers == [0, 0.6, 1, 2, -1]
    assert feature == "feature"


def test_get_tree():
    data_name = "adult"
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, "forest", "xgboost")
    booster = model.tree_iterator[0]

    tree = _get_tree_from_booster(booster)

    assert isinstance(tree, list)
    assert isinstance(tree[0], str)


def test_parse_node():
    data_name = "adult"
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, "forest", "xgboost")
    booster = model.tree_iterator[0]

    tree = _get_tree_from_booster(booster)

    leaf_str = None
    node_str = None
    for node in tree:
        if "leaf" in node and leaf_str is None:
            leaf_str = node
        elif "leaf" not in node and node_str is None:
            node_str = node

    (
        node_id,
        threshold,
        feature,
        left_child,
        right_child,
        score,
    ) = _parse_node(node_str)

    assert threshold != TREE_UNDEFINED
    assert feature != TREE_UNDEFINED
    assert left_child != TREE_LEAF
    assert right_child != TREE_LEAF
    assert score is None

    (
        node_id,
        threshold,
        feature,
        left_child,
        right_child,
        score,
    ) = _parse_node(leaf_str)

    assert threshold == TREE_UNDEFINED
    assert feature == TREE_UNDEFINED
    assert left_child == TREE_LEAF
    assert right_child == TREE_LEAF
    assert 0 <= score <= 1


def test_parse_booster():
    data_name = "adult"
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, "forest", "xgboost")
    tree = model.tree_iterator[0]

    children_left, children_right, thresholds, features, scores = parse_booster(tree)

    assert len(children_left) > 0
    assert len(children_right) > 0
    assert len(thresholds) > 0
    assert len(features) > 0
    assert len(scores) > 0
