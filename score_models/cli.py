"""Custom CLI for your score-models.

This is totally optional;
if you want to use it, though,
follow the skeleton to flesh out the CLI to your liking!
Finally, familiarize yourself with Typer,
which is the package that we use to enable this magic.
Typer's docs can be found at:

    https://typer.tiangolo.com
"""

import typer

app = typer.Typer()


@app.command()
def hello():
    """Echo the project's name."""
    typer.echo("This project's name is score-models")


@app.command()
def describe():
    """Describe the project."""
    typer.echo(
        "This project's description is me learning about score models, "
        "generating samples, and ODE nets."
    )


if __name__ == "__main__":
    app()
