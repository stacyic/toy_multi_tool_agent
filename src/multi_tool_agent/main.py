"""CLI entry point for the multi-tool agent."""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from .config import get_settings
from .core.agent import MultiToolAgent
from .core.batch_processor import BatchResult
from .core.exceptions import APIKeyMissingError
from .logging.performance_logger import PerformanceLogger
from .logging.trace_logger import TraceLogger

app = typer.Typer(
    name="multi-tool-agent",
    help="Multi-tool LangChain agent with RAG and Text-to-SQL capabilities. Use 'query' for single queries, 'batch' for multiple queries.",
    add_completion=False,
)

console = Console()


def get_agent(
    settings=None,
    enable_perf_logging: bool = False,
    enable_semantic_eval: bool = False,
    enable_human_review: bool = False,
    enable_rag_metrics: bool = False,
) -> MultiToolAgent:
    """Create and configure the agent."""
    if settings is None:
        settings = get_settings()

    if not settings.openai.api_key:
        raise APIKeyMissingError(
            "OpenAI API key is required. Set OPENAI_API_KEY environment variable or add to .env file."
        )

    # Initialize trace logger
    logger = TraceLogger(
        destination=settings.logging.destination,
        log_file=settings.logging.file,
        level=settings.logging.level,
        include_costs=settings.logging.include_costs,
    )

    # Initialize performance logger if enabled
    perf_logger = None
    if enable_perf_logging:
        perf_log_file = settings.logging.file.with_suffix(".perf.log") if settings.logging.file else None
        perf_logger = PerformanceLogger(
            log_file=perf_log_file,
            level=settings.logging.level,
            enable_console=settings.logging.destination in ("console", "both"),
            enable_file=settings.logging.destination in ("file", "both"),
            debug_sql=settings.logging.level.upper() == "DEBUG",
        )

    # Get semantic evaluation settings
    sql_val = settings.sql_validation

    # RAG metrics path
    rag_metrics_path = Path("data/rag_metrics.json") if enable_rag_metrics else None

    # Initialize agent
    agent = MultiToolAgent(
        db_path=settings.database.path,
        policies_path=settings.policies_path,
        model=settings.openai.chat_model,
        api_key=settings.openai.api_key,
        logger=logger,
        perf_logger=perf_logger,
        graceful_degradation=settings.agent.graceful_degradation,
        # Semantic evaluation options (override from CLI if provided)
        enable_semantic_eval=enable_semantic_eval or sql_val.enable_semantic_eval,
        semantic_eval_model=sql_val.semantic_eval_model or None,
        enable_human_review=enable_human_review or sql_val.enable_human_review,
        review_queue_path=sql_val.review_queue_path,
        semantic_confidence_threshold=sql_val.semantic_confidence_threshold,
        # RAG metrics options
        enable_rag_metrics=enable_rag_metrics,
        rag_metrics_path=rag_metrics_path,
    )

    return agent


async def run_query(agent: MultiToolAgent, query: str) -> str:
    """Execute a single query."""
    return await agent.execute(query)


@app.command()
def query(
    question: Optional[str] = typer.Argument(
        None,
        help="The question to ask. If not provided, enters interactive mode.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output with debug information.",
    ),
    perf: bool = typer.Option(
        False,
        "--perf",
        "-p",
        help="Enable performance logging with component-level timing.",
    ),
    eval_sql: bool = typer.Option(
        False,
        "--eval",
        "-e",
        help="Enable semantic evaluation to verify SQL correctness. Automatically records non-correct queries for human review.",
    ),
    rag_metrics: bool = typer.Option(
        False,
        "--rag-metrics",
        "-r",
        help="Enable RAG retrieval metrics tracking (similarity scores, fallback rates).",
    ),
):
    """
    Ask a question or enter interactive mode.

    Examples:
        .venv/bin/python3 -m multi_tool_agent.main query "What is the return policy?"
        .venv/bin/python3 -m multi_tool_agent.main query --perf "Show me VIP customers"
        .venv/bin/python3 -m multi_tool_agent.main query --eval "How many VIP customers?"
        .venv/bin/python3 -m multi_tool_agent.main query --rag-metrics "What are VIP benefits?"
        .venv/bin/python3 -m multi_tool_agent.main query  # Enters interactive mode
    """
    settings = get_settings()

    # Adjust log level for verbose mode
    if verbose:
        settings.logging.level = "DEBUG"

    try:
        # When semantic eval is enabled, automatically enable human review
        # to record non-correct queries
        agent = get_agent(
            settings,
            enable_perf_logging=perf,
            enable_semantic_eval=eval_sql,
            enable_human_review=eval_sql,  # Auto-enable when eval is enabled
            enable_rag_metrics=rag_metrics,
        )
    except APIKeyMissingError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if question:
        # Single query mode
        try:
            result = asyncio.run(run_query(agent, question))
            console.print(Panel(Markdown(result), title="Response", border_style="green"))

            # Display semantic evaluation results if enabled
            if eval_sql and hasattr(agent, 'sql_accessor'):
                _display_evaluation_summary(agent)

            # Display RAG metrics if enabled
            if rag_metrics:
                _display_rag_metrics(agent)

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)
    else:
        # Interactive mode
        interactive_mode(agent, show_eval=eval_sql, show_rag_metrics=rag_metrics)


def _display_evaluation_summary(agent: MultiToolAgent):
    """Display semantic evaluation summary after query execution."""
    # Check if there's review queue stats to show
    if agent.sql_accessor.review_queue:
        stats = agent.sql_accessor.get_review_queue_stats()
        if stats and stats.get("total_items", 0) > 0:
            review_table = Table(title="Human Review Queue", show_header=True)
            review_table.add_column("Metric", style="bold")
            review_table.add_column("Value", style="cyan")
            review_table.add_row("Pending reviews", str(stats.get("pending", 0)))
            review_table.add_row("Total items", str(stats.get("total_items", 0)))
            if stats.get("high_frequency_count", 0) > 0:
                review_table.add_row(
                    "High frequency issues",
                    f"[yellow]{stats['high_frequency_count']}[/yellow]"
                )
            console.print(review_table)


def _display_rag_metrics(agent: MultiToolAgent):
    """Display RAG retrieval metrics after query execution."""
    if not agent.rag_metrics_tracker:
        return

    # Get the most recent retrieval result for detailed per-chunk display
    tracker = agent.rag_metrics_tracker
    if tracker.results:
        last_result = tracker.results[-1]

        # Per-chunk details table with citations
        chunk_table = Table(title="Retrieved Chunks (Latest Query)", show_header=True)
        chunk_table.add_column("#", style="dim", width=3)
        chunk_table.add_column("Section", style="bold")
        chunk_table.add_column("Lines", style="dim", justify="center")
        chunk_table.add_column("Score", justify="right")
        chunk_table.add_column("Relevance", justify="center")

        # Use chunk_details if available, otherwise fall back to basic info
        if last_result.chunk_details:
            for i, chunk in enumerate(last_result.chunk_details, 1):
                # Color code by relevance
                if chunk.score >= 0.8:
                    score_str = f"[green]{chunk.score:.4f}[/green]"
                    relevance = "[green]High[/green]"
                elif chunk.score >= 0.6:
                    score_str = f"[cyan]{chunk.score:.4f}[/cyan]"
                    relevance = "[cyan]Medium[/cyan]"
                elif chunk.score >= 0.4:
                    score_str = f"[yellow]{chunk.score:.4f}[/yellow]"
                    relevance = "[yellow]Low[/yellow]"
                else:
                    score_str = f"[red]{chunk.score:.4f}[/red]"
                    relevance = "[red]Poor[/red]"

                # Section path (truncate if needed)
                section = chunk.section_path or chunk.source
                section_display = section[:35] + "..." if len(section) > 35 else section

                # Line numbers
                if chunk.line_start and chunk.line_end:
                    lines_str = f"{chunk.line_start}-{chunk.line_end}"
                else:
                    lines_str = "-"

                chunk_table.add_row(str(i), section_display, lines_str, score_str, relevance)
        else:
            # Fallback to basic display
            for i, (score, source) in enumerate(zip(last_result.similarity_scores, last_result.sources), 1):
                if score >= 0.8:
                    score_str = f"[green]{score:.4f}[/green]"
                    relevance = "[green]High[/green]"
                elif score >= 0.6:
                    score_str = f"[cyan]{score:.4f}[/cyan]"
                    relevance = "[cyan]Medium[/cyan]"
                elif score >= 0.4:
                    score_str = f"[yellow]{score:.4f}[/yellow]"
                    relevance = "[yellow]Low[/yellow]"
                else:
                    score_str = f"[red]{score:.4f}[/red]"
                    relevance = "[red]Poor[/red]"

                source_display = source[:35] + "..." if len(source) > 35 else source
                chunk_table.add_row(str(i), source_display, "-", score_str, relevance)

        # Add summary row
        if last_result.similarity_scores:
            avg = last_result.avg_similarity
            chunk_table.add_row("", "[dim]Average[/dim]", "", f"[dim]{avg:.4f}[/dim]", "")

        console.print(chunk_table)

        # Source citation info
        console.print(f"[dim]Source: policies.md[/dim]")
        console.print(f"[dim]Query: {last_result.query[:60]}{'...' if len(last_result.query) > 60 else ''}[/dim]")
        console.print(f"[dim]Latency: {last_result.latency_ms:.1f}ms | Fallback: {last_result.from_fallback}[/dim]")

    # Aggregate metrics (if multiple queries in session)
    metrics = tracker.get_metrics().to_dict()
    if metrics.get("total_queries", 0) > 1:
        console.print()
        agg_table = Table(title="Session Aggregate Metrics", show_header=True)
        agg_table.add_column("Metric", style="bold")
        agg_table.add_column("Value", style="cyan")

        agg_table.add_row("Total queries", str(metrics.get("total_queries", 0)))
        agg_table.add_row("Total chunks", str(metrics.get("total_chunks_retrieved", 0)))
        agg_table.add_row("Avg similarity", f"{metrics.get('avg_similarity', 0):.4f}")
        agg_table.add_row("High relevance rate", f"{metrics.get('high_relevance_rate_pct', 0):.1f}%")
        agg_table.add_row("Fallback rate", f"{metrics.get('fallback_rate_pct', 0):.1f}%")

        console.print(agg_table)


def interactive_mode(agent: MultiToolAgent, show_eval: bool = False, show_rag_metrics: bool = False):
    """Run interactive REPL mode."""
    status_parts = []
    if show_eval:
        status_parts.append("semantic eval")
    if show_rag_metrics:
        status_parts.append("RAG metrics")
    status_str = f" [dim]({', '.join(status_parts)} enabled)[/dim]" if status_parts else ""

    console.print(
        Panel(
            f"[bold]Multi-Tool Agent[/bold]{status_str}\n\n"
            "Ask questions about policies, customers, products, and orders.\n"
            "Type 'exit' or 'quit' to leave, 'help' for assistance.",
            title="Welcome",
            border_style="blue",
        )
    )

    # Show schema on startup
    console.print("\n[dim]Database tables: customers, products, orders, order_items[/dim]\n")

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")

            if not question.strip():
                continue

            if question.lower() in ("exit", "quit", "q"):
                # Export RAG metrics on exit if enabled
                if show_rag_metrics:
                    agent.export_rag_metrics()
                    console.print("[dim]RAG metrics exported to data/rag_metrics.json[/dim]")
                console.print("[dim]Goodbye![/dim]")
                break

            if question.lower() == "help":
                show_help()
                continue

            if question.lower() == "schema":
                console.print(Panel(agent.get_schema_description(), title="Database Schema"))
                continue

            if question.lower() == "review-stats":
                _display_evaluation_summary(agent)
                continue

            if question.lower() == "rag-stats":
                _display_rag_metrics(agent)
                continue

            # Execute query
            with console.status("[bold green]Thinking...[/bold green]"):
                result = asyncio.run(run_query(agent, question))

            console.print(Panel(Markdown(result), title="Assistant", border_style="green"))

            # Show evaluation summary if enabled
            if show_eval:
                _display_evaluation_summary(agent)

            # Show RAG metrics if enabled
            if show_rag_metrics:
                _display_rag_metrics(agent)

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'exit' to quit.[/dim]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


def show_help():
    """Show help information."""
    help_text = """
## Commands
- **exit**, **quit**, **q**: Exit the program
- **help**: Show this help message
- **schema**: Show database schema
- **review-stats**: Show SQL evaluation review queue stats
- **rag-stats**: Show RAG retrieval metrics

## Example Questions

### Policy Questions
- What is the return policy?
- What are the VIP customer benefits?
- How long is the warranty on electronics?

### Data Questions
- List all products under $100
- How many orders were placed last month?
- Show me the top 5 customers by order count

### Combined Questions
- How many VIP customers do we have?
- Which customers are eligible for free shipping?
"""
    console.print(Panel(Markdown(help_text), title="Help", border_style="blue"))


@app.command("init-db")
def init_db(
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        "-d",
        help="Directory containing CSV files.",
    ),
    output: Path = typer.Option(
        Path("data/store.db"),
        "--output",
        "-o",
        help="Output database path.",
    ),
):
    """
    Initialize the database from CSV files.

    Expects customers.csv, products.csv, orders.csv, and order_items.csv
    in the data directory.
    """
    from scripts.init_db import initialize_database

    try:
        initialize_database(output, data_dir)
        console.print(f"[green]Database initialized at {output}[/green]")
    except Exception as e:
        console.print(f"[red]Error initializing database:[/red] {e}")
        raise typer.Exit(1)


@app.command("test-connection")
def test_connection():
    """Test the OpenAI API connection."""
    settings = get_settings()

    if not settings.openai.api_key:
        console.print("[red]Error:[/red] OPENAI_API_KEY not set")
        raise typer.Exit(1)

    console.print("[dim]Testing OpenAI API connection...[/dim]")

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.openai.chat_model,
            api_key=settings.openai.api_key,
        )
        response = asyncio.run(llm.ainvoke("Say 'Connection successful' in 3 words or less."))
        console.print(f"[green]Success:[/green] {response.content}")
    except Exception as e:
        console.print(f"[red]Connection failed:[/red] {e}")
        raise typer.Exit(1)


@app.command("batch")
def batch_query(
    queries: Optional[List[str]] = typer.Argument(
        None,
        help="List of queries to process. If not provided, reads from --file or stdin.",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="JSON file containing list of queries (array of strings).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for results. If not specified, prints to console.",
    ),
    max_concurrent: int = typer.Option(
        5,
        "--concurrent",
        "-c",
        help="Maximum concurrent query executions.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output with per-query metrics.",
    ),
    perf: bool = typer.Option(
        False,
        "--perf",
        "-p",
        help="Enable performance logging with component-level timing.",
    ),
    show_metrics: bool = typer.Option(
        True,
        "--metrics/--no-metrics",
        help="Show batch metrics summary.",
    ),
):
    """
    Process multiple queries in batch mode with optimized API calls.

    Examples:
        # From command line arguments
        .venv/bin/python3 -m multi_tool_agent.main batch "What is the return policy?" "How many VIP customers?"

        # From JSON file with performance logging
        .venv/bin/python3 -m multi_tool_agent.main batch --file queries.json --output results.json --perf

        # From stdin (pipe)
        echo '["Q1", "Q2"]' | .venv/bin/python3 -m multi_tool_agent.main batch
    """
    settings = get_settings()

    if verbose:
        settings.logging.level = "DEBUG"

    try:
        agent = get_agent(settings, enable_perf_logging=perf)
    except APIKeyMissingError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Collect queries from various sources
    query_list: List[str] = []

    if queries:
        query_list = list(queries)
    elif file:
        if not file.exists():
            console.print(f"[red]Error:[/red] File not found: {file}")
            raise typer.Exit(1)
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    query_list = [str(q) for q in data]
                else:
                    console.print("[red]Error:[/red] JSON file must contain an array of query strings")
                    raise typer.Exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON file: {e}")
            raise typer.Exit(1)
    elif not sys.stdin.isatty():
        # Read from stdin
        try:
            stdin_data = sys.stdin.read().strip()
            data = json.loads(stdin_data)
            if isinstance(data, list):
                query_list = [str(q) for q in data]
            else:
                console.print("[red]Error:[/red] Stdin must contain a JSON array of query strings")
                raise typer.Exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON from stdin: {e}")
            raise typer.Exit(1)
    else:
        console.print("[red]Error:[/red] No queries provided. Use arguments, --file, or pipe JSON to stdin.")
        raise typer.Exit(1)

    if not query_list:
        console.print("[yellow]Warning:[/yellow] No queries to process.")
        raise typer.Exit(0)

    console.print(f"\n[bold]Processing {len(query_list)} queries...[/bold]\n")

    # Execute batch
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Processing batch of {len(query_list)} queries...",
                total=None,
            )
            result = asyncio.run(
                agent.execute_batch(query_list, max_concurrent=max_concurrent)
            )
            progress.update(task, completed=True)
    except Exception as e:
        console.print(f"[red]Error during batch execution:[/red] {e}")
        raise typer.Exit(1)

    # Display results
    display_batch_results(result, verbose=verbose, show_metrics=show_metrics)

    # Write to output file if specified
    if output:
        try:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            console.print(f"\n[green]Results written to:[/green] {output}")
        except Exception as e:
            console.print(f"[red]Error writing output file:[/red] {e}")
            raise typer.Exit(1)


def display_batch_results(
    result: BatchResult,
    verbose: bool = False,
    show_metrics: bool = True,
):
    """Display batch processing results."""
    # Results table
    for i, qr in enumerate(result.results):
        status = "[green]✓[/green]" if qr.success else "[red]✗[/red]"
        header = f"{status} Query {i + 1}: {qr.query[:50]}{'...' if len(qr.query) > 50 else ''}"

        if qr.success:
            console.print(Panel(
                Markdown(qr.response),
                title=header,
                border_style="green" if qr.success else "red",
            ))
        else:
            console.print(Panel(
                f"[red]Error:[/red] {qr.error}\n\n{qr.response}",
                title=header,
                border_style="red",
            ))

        if verbose and qr.metrics:
            metrics_table = Table(show_header=False, box=None, padding=(0, 1))
            metrics_table.add_column("Metric", style="dim")
            metrics_table.add_column("Value", style="cyan")
            metrics_table.add_row("Total time", f"{qr.metrics.total_time_ms:.0f}ms")
            if qr.metrics.routing_time_ms > 0:
                metrics_table.add_row("Routing", f"{qr.metrics.routing_time_ms:.0f}ms")
            if qr.metrics.policy_time_ms > 0:
                metrics_table.add_row("Policy search", f"{qr.metrics.policy_time_ms:.0f}ms")
            if qr.metrics.sql_time_ms > 0:
                metrics_table.add_row("SQL execution", f"{qr.metrics.sql_time_ms:.0f}ms")
                metrics_table.add_row("SQL attempts", str(qr.metrics.sql_attempts))
            if qr.metrics.synthesis_time_ms > 0:
                metrics_table.add_row("Synthesis", f"{qr.metrics.synthesis_time_ms:.0f}ms")
            if qr.metrics.pii_masked:
                pii_str = ", ".join(f"{k}={v}" for k, v in qr.metrics.pii_masked.items())
                metrics_table.add_row("PII masked", pii_str)
            console.print(metrics_table)
        console.print()

    # Batch metrics summary
    if show_metrics:
        metrics = result.batch_metrics
        summary_table = Table(title="Batch Summary", show_header=True)
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value", style="cyan")

        summary_table.add_row("Total queries", str(metrics.total_queries))
        summary_table.add_row(
            "Successful",
            f"[green]{metrics.successful_queries}[/green]"
        )
        if metrics.failed_queries > 0:
            summary_table.add_row(
                "Failed",
                f"[red]{metrics.failed_queries}[/red]"
            )
        summary_table.add_row("Total time", f"{metrics.total_time_ms:.0f}ms")
        summary_table.add_row("Avg time/query", f"{metrics.avg_time_per_query_ms:.0f}ms")
        if metrics.policy_calls_saved > 0:
            summary_table.add_row(
                "Policy calls saved",
                f"[green]{metrics.policy_calls_saved}[/green]"
            )

        console.print(summary_table)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
