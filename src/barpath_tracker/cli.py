"""Console script for barpath_tracker."""


import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("barpath-tracker")
    click.echo("=" * len("barpath-tracker"))
    click.echo("Automatic video annotation for oly weightlifting")


if __name__ == "__main__":
    main()
    
