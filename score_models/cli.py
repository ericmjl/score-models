"""CLI for score_models."""
import typer

from score_models.example import hello_world

app = typer.Typer()


@app.command()
def hello():
    """Hello command"""
    typer.echo(hello_world())


if __name__ == "__main__":
    app()
