"""Command line interface for histo-omics-lite."""

from __future__ import annotations

import typer

from . import data as data_cmd
from . import eval as eval_cmd
from . import infer as infer_cmd
from . import train as train_cmd

app = typer.Typer(help="CLI for histo-omics-lite pipelines", no_args_is_help=True)

app.add_typer(data_cmd.app, name="data")
app.add_typer(train_cmd.app, name="train")
app.add_typer(eval_cmd.app, name="eval")
app.add_typer(infer_cmd.app, name="infer")

__all__ = ["app"]
