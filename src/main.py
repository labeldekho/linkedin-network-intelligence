"""
LinkedIn Network Intelligence CLI

Command-line interface for processing LinkedIn data exports.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Initialize console for rich output
console = Console()


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging with rich handler."""
    handlers = [RichHandler(console=console, show_path=False)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=handlers,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """LinkedIn Network Intelligence - Transform your network data into insights."""
    ctx.ensure_object(dict)

    if verbose:
        ctx.obj["log_level"] = "DEBUG"
    elif quiet:
        ctx.obj["log_level"] = "WARNING"
    else:
        ctx.obj["log_level"] = "INFO"

    setup_logging(ctx.obj["log_level"])


@cli.command()
@click.option(
    "--input", "-i",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing LinkedIn export files",
)
@click.option(
    "--output", "-o",
    "output_dir",
    default="./outputs",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for reports",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama", "vllm"]),
    default="anthropic",
    help="LLM provider to use",
)
@click.option(
    "--model",
    default=None,
    help="Model name (uses provider default if not specified)",
)
@click.option(
    "--format", "-f",
    "formats",
    multiple=True,
    type=click.Choice(["csv", "markdown", "json"]),
    default=["csv", "markdown", "json"],
    help="Output formats to generate",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Skip LLM enrichment (use cached/mock responses)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable response caching",
)
@click.pass_context
def process(
    ctx: click.Context,
    input_dir: str,
    output_dir: str,
    provider: str,
    model: Optional[str],
    formats: tuple[str, ...],
    dry_run: bool,
    no_cache: bool,
) -> None:
    """Process LinkedIn export and generate relationship intelligence reports."""
    from src.pipeline.ingest import load_linkedin_export
    from src.pipeline.normalize import normalize_network
    from src.models.relationship import RelationshipStrengthCalculator
    from src.models.reciprocity import ReciprocityLedger
    from src.pipeline.outputs import generate_outputs
    from src.utils.config import load_config

    config = load_config()

    console.print("\n[bold blue]LinkedIn Network Intelligence[/bold blue]")
    console.print("=" * 50)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Load data
        task = progress.add_task("Loading LinkedIn export...", total=None)
        try:
            export = load_linkedin_export(input_dir, require_messages=False)
            progress.update(task, completed=True)
            console.print(f"  [green]✓[/green] Loaded {len(export.connections)} connections, {len(export.messages)} messages")
        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"  [red]✗[/red] Failed to load data: {e}")
            sys.exit(1)

        # Normalize data
        task = progress.add_task("Normalizing network data...", total=None)
        snapshot = normalize_network(export, thread_break_days=config.messages.thread_break_days)
        progress.update(task, completed=True)
        console.print(f"  [green]✓[/green] Resolved {len(snapshot.people)} unique people")

        # Calculate relationship strength
        task = progress.add_task("Calculating relationship strengths...", total=None)
        calculator = RelationshipStrengthCalculator(
            decay_half_life_days=config.relationship.decay_half_life_days,
            minimum_strength=config.relationship.minimum_strength,
            depth_multiplier_max=config.relationship.depth_multiplier_max,
        )
        snapshot = calculator.calculate_network_strengths(snapshot)
        progress.update(task, completed=True)
        console.print(f"  [green]✓[/green] Average strength: {snapshot.avg_relationship_strength:.2f}")

        # Calculate reciprocity
        task = progress.add_task("Building reciprocity ledger...", total=None)
        ledger = ReciprocityLedger()
        snapshot, balances = ledger.calculate_network_balances(snapshot)
        reciprocity_summary = ledger.get_summary(balances)
        progress.update(task, completed=True)

        # LLM Enrichment
        analysis_results = {"reciprocity_summary": reciprocity_summary}

        if not dry_run:
            task = progress.add_task("Enriching with LLM analysis...", total=100)

            async def run_enrichment():
                from src.pipeline.enrich import enrich_network

                def progress_cb(current, total):
                    progress.update(task, completed=int(current / total * 100))

                return await enrich_network(
                    snapshot,
                    provider_name=provider,
                    model=model,
                    cache_enabled=not no_cache,
                )

            try:
                enriched_snapshot, enrichment_results = asyncio.run(run_enrichment())
                snapshot = enriched_snapshot
                analysis_results.update(enrichment_results)
                progress.update(task, completed=100)
                console.print(f"  [green]✓[/green] Network archetype: {snapshot.network_archetype or 'Unknown'}")
            except Exception as e:
                progress.update(task, completed=100)
                console.print(f"  [yellow]![/yellow] LLM enrichment failed: {e}")
                logging.debug(f"Enrichment error: {e}", exc_info=True)
        else:
            console.print("  [yellow]![/yellow] Skipping LLM enrichment (dry-run mode)")

        # Generate outputs
        task = progress.add_task("Generating reports...", total=None)
        output_files = generate_outputs(
            snapshot,
            analysis_results,
            output_dir=output_dir,
            formats=list(formats),
        )
        progress.update(task, completed=True)

    # Print summary
    console.print("\n[bold]Reports Generated:[/bold]")
    for report_type, files in output_files.items():
        for fmt, path in files.items():
            console.print(f"  • {report_type}.{fmt}: [cyan]{path}[/cyan]")

    # Print top relationships
    console.print("\n[bold]Top 5 Relationships:[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Company")
    table.add_column("Strength", justify="right")

    for rel in snapshot.get_top_relationships(5):
        table.add_row(
            rel.person.display_name,
            rel.person.company or "Unknown",
            f"{rel.strength_score:.2f}",
        )

    console.print(table)
    console.print()


@cli.command("warm-paths")
@click.option(
    "--target", "-t",
    required=True,
    help="Target company name",
)
@click.option(
    "--input", "-i",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing LinkedIn export files",
)
@click.option(
    "--output", "-o",
    "output_dir",
    default="./outputs",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for reports",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama", "vllm"]),
    default="anthropic",
    help="LLM provider to use",
)
@click.option(
    "--model",
    default=None,
    help="Model name",
)
@click.option(
    "--role",
    default=None,
    help="Target role/department (optional)",
)
@click.option(
    "--reason",
    default=None,
    help="Reason for seeking introduction",
)
@click.option(
    "--max-candidates",
    default=10,
    help="Maximum candidates to return",
)
@click.pass_context
def warm_paths(
    ctx: click.Context,
    target: str,
    input_dir: str,
    output_dir: str,
    provider: str,
    model: Optional[str],
    role: Optional[str],
    reason: Optional[str],
    max_candidates: int,
) -> None:
    """Find warm introduction paths to a target company."""
    from src.pipeline.ingest import load_linkedin_export
    from src.pipeline.normalize import normalize_network
    from src.models.relationship import RelationshipStrengthCalculator
    from src.models.warm_paths import WarmPathFinder
    from src.pipeline.outputs import OutputGenerator
    from src.utils.config import load_config

    config = load_config()

    console.print(f"\n[bold blue]Finding Warm Paths to {target}[/bold blue]")
    console.print("=" * 50)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load and process data
        task = progress.add_task("Loading and processing data...", total=None)
        try:
            export = load_linkedin_export(input_dir, require_messages=False)
            snapshot = normalize_network(export)

            calculator = RelationshipStrengthCalculator(
                decay_half_life_days=config.relationship.decay_half_life_days,
            )
            snapshot = calculator.calculate_network_strengths(snapshot)
            progress.update(task, completed=True)
        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

        # Find warm paths
        task = progress.add_task("Analyzing potential bridges...", total=None)

        async def find_paths():
            finder = WarmPathFinder(
                provider_name=provider,
                model=model,
                max_candidates=max_candidates,
                min_bridge_strength=config.warm_paths.min_bridge_strength,
            )
            return await finder.find_warm_paths(
                snapshot,
                target_company=target,
                target_role=role,
                reason=reason,
            )

        try:
            candidates = asyncio.run(find_paths())
            progress.update(task, completed=True)
        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"[red]Analysis failed: {e}[/red]")
            logging.debug(f"Warm path error: {e}", exc_info=True)
            sys.exit(1)

    # Generate report
    generator = OutputGenerator(output_dir=output_dir, formats=["markdown"])
    from src.models.warm_paths import WarmPathFinder as WPF
    finder = WPF()
    summary = finder.get_summary(candidates, target)
    output_files = generator.generate_warm_paths(candidates, target, summary)

    # Print results
    if not candidates:
        console.print(f"\n[yellow]No warm paths found to {target}[/yellow]")
        console.print("Consider expanding your network in that industry.")
        return

    console.print(f"\n[bold]Found {len(candidates)} potential paths:[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Name")
    table.add_column("Company")
    table.add_column("Connection", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Viable", justify="center")

    for i, c in enumerate(candidates[:10], 1):
        table.add_row(
            str(i),
            c.person_name,
            c.company or "Unknown",
            c.connection_type,
            f"{c.composite_score:.2f}",
            "[green]Yes[/green]" if c.is_viable_bridge else "[dim]No[/dim]",
        )

    console.print(table)

    console.print(f"\n[dim]Full report: {output_files.get('markdown', 'N/A')}[/dim]")


@cli.command()
@click.option(
    "--input", "-i",
    "input_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing LinkedIn export files",
)
@click.pass_context
def stats(ctx: click.Context, input_dir: str) -> None:
    """Show quick statistics about your LinkedIn export."""
    from src.pipeline.ingest import load_linkedin_export

    console.print("\n[bold blue]LinkedIn Export Statistics[/bold blue]")
    console.print("=" * 50)

    try:
        export = load_linkedin_export(input_dir, require_messages=False)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Source:[/bold] {input_dir}")
    console.print(f"\n[bold]Files loaded:[/bold]")
    for f in export.loaded_files:
        console.print(f"  • {f}")

    if export.skipped_files:
        console.print(f"\n[dim]Files not found:[/dim]")
        for f in export.skipped_files:
            console.print(f"  • {f}")

    console.print(f"\n[bold]Data Summary:[/bold]")
    table = Table(show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("Connections", str(len(export.connections)))
    table.add_row("Messages", str(len(export.messages)))
    table.add_row("Endorsements given", str(len(export.endorsements_given)))
    table.add_row("Endorsements received", str(len(export.endorsements_received)))
    table.add_row("Recommendations given", str(len(export.recommendations_given)))
    table.add_row("Recommendations received", str(len(export.recommendations_received)))

    console.print(table)

    # Company distribution
    companies: dict[str, int] = {}
    for conn in export.connections:
        if conn.company:
            companies[conn.company] = companies.get(conn.company, 0) + 1

    if companies:
        top_companies = sorted(companies.items(), key=lambda x: x[1], reverse=True)[:10]

        console.print(f"\n[bold]Top Companies in Network:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Company")
        table.add_column("Connections", justify="right")

        for company, count in top_companies:
            table.add_row(company, str(count))

        console.print(table)

    console.print()


@cli.command()
@click.pass_context
def version(ctx: click.Context) -> None:
    """Show version information."""
    from src import __version__

    console.print(f"LinkedIn Network Intelligence v{__version__}")


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
