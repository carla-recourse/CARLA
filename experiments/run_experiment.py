# flake8: noqa
import os

from carla import log

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
from typing import Dict, Optional

import numpy as np
import yaml
from tensorflow import Graph, Session

from carla.data.api import Data
from carla.data.catalog import DataCatalog
from carla.evaluation import Benchmark
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods import *
from carla.recourse_methods.api import RecourseMethod


def save_result(result: pd.DataFrame, alt_path: Optional[str]) -> None:
    data_home = os.environ.get("CF_DATA", os.path.join("~", "carla", "results"))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    path = os.path.join(data_home, "results.csv") if alt_path is None else alt_path

    result.to_csv(path, index=False)


def load_setup() -> Dict:
    with open("experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)

    return setup_catalog["recourse_methods"]


def initialize_recourse_method(
    method: str,
    mlmodel: MLModel,
    data: Data,
    data_name: str,
    model_type: str,
    setup: Dict,
    sess: Session = None,
) -> RecourseMethod:
    if method not in setup.keys():
        raise KeyError("Method not in experimental setup")

    hyperparams = setup[method]["hyperparams"]
    if method == "ar":
        coeffs, intercepts = None, None
        if model_type == "linear":
            # get weights and bias of linear layer for negative class 0
            coeffs = mlmodel.raw_model.layers[0].get_weights()[0][:, 0]
            intercepts = np.array(mlmodel.raw_model.layers[0].get_weights()[1][0])

        ar = ActionableRecourse(mlmodel, hyperparams, coeffs, intercepts)
        act_set = ar.action_set

        # some datasets need special configuration for possible actions
        if data_name == "give_me_some_credit":
            act_set["NumberOfTimes90DaysLate"].mutable = False
            act_set["NumberOfTimes90DaysLate"].actionable = False
            act_set["NumberOfTime60-89DaysPastDueNotWorse"].mutable = False
            act_set["NumberOfTime60-89DaysPastDueNotWorse"].actionable = False

        ar.action_set = act_set

        return ar
    elif "cem" in method:
        hyperparams["data_name"] = data_name
        return CEM(sess, mlmodel, hyperparams)
    elif method == "clue":
        hyperparams["data_name"] = data_name
        return Clue(data, mlmodel, hyperparams)
    elif method == "dice":
        return Dice(mlmodel, hyperparams)
    elif "face" in method:
        return Face(mlmodel, hyperparams)
    elif method == "gs":
        return GrowingSpheres(mlmodel)
    elif method == "revise":
        hyperparams["data_name"] = data_name
        # variable input layer dimension is first time here available
        hyperparams["vae_params"]["layers"] = [
            len(mlmodel.feature_input_order)
        ] + hyperparams["vae_params"]["layers"]
        return Revise(mlmodel, data, hyperparams)
    elif "wachter" in method:
        return Wachter(mlmodel, hyperparams)
    else:
        raise ValueError("Recourse method not known")


parser = argparse.ArgumentParser(description="Run experiments from paper")
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    default=["adult", "compas", "give_me_some_credit"],
    choices=["adult", "compas", "give_me_some_credit"],
    help="Datasets for experiment",
)
parser.add_argument(
    "-t",
    "--type",
    nargs="*",
    default=["ann", "linear"],
    choices=["ann", "linear"],
    help="Model type for experiment",
)
parser.add_argument(
    "-r",
    "--recourse_method",
    nargs="*",
    default=[
        "dice",
        "ar",
        "cem",
        "cem-vae",
        "clue",
        "face_knn",
        "face_epsilon",
        "gs",
        "revise",
        "wachter",
    ],
    choices=[
        "dice",
        "ar",
        "cem",
        "cem-vae",
        "clue",
        "face_knn",
        "face_epsilon",
        "gs",
        "revise",
        "wachter",
    ],
    help="Recourse methods for experiment",
)
parser.add_argument(
    "-n",
    "--number_of_samples",
    type=int,
    default=100,
    help="Number of instances per dataset",
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=None,
    help="Save path for the output csv. If None, the output is written to the cache.",
)
args = parser.parse_args()
setup = load_setup()

results = pd.DataFrame()

path = args.path

session_models = ["cem", "cem-vae"]
torch_methods = ["clue", "wachter", "revise"]
for rm in args.recourse_method:
    backend = "tensorflow"
    if rm in torch_methods:
        backend = "pytorch"
    for data_name in args.dataset:
        dataset = DataCatalog(data_name)
        for model_type in args.type:
            log.info("=====================================")
            log.info("Recourse method: {}".format(rm))
            log.info("Dataset: {}".format(data_name))
            log.info("Model type: {}".format(model_type))

            if rm in session_models:
                graph = Graph()
                with graph.as_default():
                    ann_sess = Session()
                    with ann_sess.as_default():
                        mlmodel_sess = MLModelCatalog(dataset, model_type, backend)

                        factuals_sess = predict_negative_instances(
                            mlmodel_sess, dataset
                        )
                        factuals_sess = factuals_sess.iloc[: args.number_of_samples]
                        factuals_sess = factuals_sess.reset_index(drop=True)

                        recourse_method_sess = initialize_recourse_method(
                            rm,
                            mlmodel_sess,
                            dataset,
                            data_name,
                            model_type,
                            setup,
                            sess=ann_sess,
                        )

                        df_benchmark = Benchmark(
                            mlmodel_sess, recourse_method_sess, factuals_sess
                        ).run_benchmark()
            else:
                mlmodel = MLModelCatalog(dataset, model_type, backend)

                factuals = predict_negative_instances(mlmodel, dataset)
                factuals = factuals.iloc[: args.number_of_samples]
                factuals = factuals.reset_index(drop=True)

                if rm == "dice":
                    mlmodel.use_pipeline = True

                recourse_method = initialize_recourse_method(
                    rm, mlmodel, dataset, data_name, model_type, setup
                )

                df_benchmark = Benchmark(
                    mlmodel, recourse_method, factuals
                ).run_benchmark()

            df_benchmark["Recourse_Method"] = rm
            df_benchmark["Dataset"] = data_name
            df_benchmark["ML_Model"] = model_type
            df_benchmark = df_benchmark[
                [
                    "Recourse_Method",
                    "Dataset",
                    "ML_Model",
                    "Distance_1",
                    "Distance_2",
                    "Distance_3",
                    "Distance_4",
                    "Constraint_Violation",
                    "Redundancy",
                    "y-Nearest-Neighbours",
                    "Success_Rate",
                    "Average_Time",
                ]
            ]

            results = pd.concat([results, df_benchmark], axis=0)
            log.info("=====================================")

save_result(results, path)
