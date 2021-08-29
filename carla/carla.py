#!/usr/bin/env python3

import click

from carla import DataCatalog, MLModelCatalog
from carla.recourse_methods import *


@click.group()
def cli():
    click.echo("Hello World")


@cli.command()
@click.option(
    "--data",
    "data_name",
    default="adult",
    help="The dataset to generate counterfactuals on",
    type=click.Choice(["adult", "compas"], case_sensitive=False),
)
@click.option(
    "--model",
    "model_name",
    required=True,
    default="ann",
    help="The black-box model to use",
    type=click.Choice(["ann", "lr"], case_sensitive=False),
)
@click.option(
    "--method",
    "method_name",
    required=True,
    default="gs",
    help="The counterfactual method to run",
    type=click.Choice(["gs", "face"], case_sensitive=False),
)
@click.option(
    "--sample-size",
    required=True,
    default=5,
    help="The number of factual samples from the dataset",
)
def run(data_name, method_name, model_name, sample_size):
    click.echo("Run a single counterfactual method")
    dataset = DataCatalog(data_name)
    model = MLModelCatalog(dataset, model_name)

    if method_name == "gs":
        method = GrowingSpheres(model)
    elif method_name == "face":
        method = Face(model)
    else:
        raise ValueError(f"Recourse model {model_name} unknown.")

    factuals = dataset.raw.sample(sample_size)
    counterfactuals = method.get_counterfactuals(factuals)
    click.echo(counterfactuals)


@cli.command()
def benchmark():
    click.echo("Benchmark")


if __name__ == "__main__":
    cli()
