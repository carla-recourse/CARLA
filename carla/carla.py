#!/usr/bin/env python3

import click


@click.group()
def cli():
    click.echo("Hello World")


@cli.command()
@click.option(
    "--data",
    default="ann",
    help="The dataset to generate counterfactuals on",
    type=click.Choice(["ann", "compas"], case_sensitive=False),
)
@click.option(
    "--method",
    required=True,
    default="gs",
    help="The counterfactual method to run",
    type=click.Choice(["gs", "face"], case_sensitive=False),
)
def run(method):
    click.echo("Run a single counterfactual method")


@cli.command()
def benchmark():
    click.echo("Benchmark")


if __name__ == "__main__":
    cli()
