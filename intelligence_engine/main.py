"""
main.py
-------
Application execution entry point for the Automated Enterprise Market
Intelligence & Fact-Checking Engine.

Usage
-----
    # From the repository root with a valid .env file:
    python -m intelligence_engine.main

    # Or directly:
    python intelligence_engine/main.py

What this script does
---------------------
1. Initialises the configuration and validates all required environment
   variables.
2. Streams the LangGraph pipeline step-by-step, printing rich terminal output
   at every node transition so you can observe the self-correction loops and
   circuit breaker in real-time.
3. Prints the final validated JSON report cleanly at the end.

Terminal output key
-------------------
  [▶ NODE]     — A graph node is starting execution
  [✓ NODE]     — A graph node completed successfully
  [↻ LOOP]     — The critic routed back to the generator for corrections
  [⚡ BREAK]   — The circuit breaker fired; using best-effort draft
  [✅ DONE]    — Pipeline completed; final report follows
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from typing import Any, Dict, Optional

from intelligence_engine.config import get_settings, logger
from intelligence_engine.graph import app
from intelligence_engine.schema import MarketIntelligenceReport, initial_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The research prompt used for the interactive demo run.
DEMO_QUERY = (
    "Provide a comprehensive market intelligence analysis of NVIDIA Corporation. "
    "Cover their AI chip dominance (H100/H200 GPU revenue), the role of the CUDA "
    "software ecosystem as a competitive moat, key risk factors including US export "
    "controls and hyperscaler in-house silicon programs, and the company's current "
    "market capitalisation trajectory heading into 2025."
)

# ANSI colour codes for terminal readability.
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    banner = f"""
{_BOLD}{_CYAN}
╔══════════════════════════════════════════════════════════════════════╗
║   Automated Enterprise Market Intelligence & Fact-Checking Engine    ║
║   Powered by LangGraph · Qdrant · Tavily · OpenAI GPT-4.1-mini     ║
╚══════════════════════════════════════════════════════════════════════╝
{_RESET}"""
    print(banner)


def _print_section(title: str, char: str = "─", width: int = 70) -> None:
    bar = char * width
    print(f"\n{_BOLD}{bar}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{bar}{_RESET}")


def _print_node_start(node_name: str) -> None:
    print(f"\n{_CYAN}{_BOLD}[▶ NODE]{_RESET}  {node_name.upper()}")


def _print_node_end(node_name: str, elapsed: float) -> None:
    print(f"{_GREEN}{_BOLD}[✓ NODE]{_RESET}  {node_name.upper()} completed in {elapsed:.2f}s")


def _print_loop_info(loop_counter: int, feedback_preview: str) -> None:
    print(
        f"\n{_YELLOW}{_BOLD}[↻ LOOP {loop_counter}]{_RESET}  "
        f"Critic rejected draft — routing back to generator.\n"
        f"  Feedback preview: {feedback_preview[:120]}{'...' if len(feedback_preview) > 120 else ''}"
    )


def _print_circuit_breaker(loop_counter: int) -> None:
    print(
        f"\n{_RED}{_BOLD}[⚡ CIRCUIT BREAKER]{_RESET}  "
        f"Maximum correction loops ({loop_counter}) reached. "
        f"Publishing best-effort draft."
    )


def _print_final_report(report_dict: Dict[str, Any]) -> None:
    _print_section("✅  FINAL VALIDATED MARKET INTELLIGENCE REPORT", "═")

    try:
        report = MarketIntelligenceReport.model_validate(report_dict)
    except Exception as exc:
        print(f"{_RED}WARNING: Final report failed re-validation: {exc}{_RESET}")
        print(json.dumps(report_dict, indent=2))
        return

    print(f"\n{_BOLD}Company:{_RESET}       {report.company_name}")
    print(f"{_BOLD}Valuation:{_RESET}     {report.market_cap_or_valuation}")

    print(f"\n{_BOLD}Core Revenue Drivers:{_RESET}")
    for i, driver in enumerate(report.core_revenue_drivers, 1):
        print(f"  {i}. {driver}")

    print(f"\n{_BOLD}Risk Factors:{_RESET}")
    for i, risk in enumerate(report.risk_factors, 1):
        print(f"  {i}. {risk}")

    print(f"\n{_BOLD}Sources:{_RESET}")
    for source in report.sources:
        print(f"  • {source}")

    _print_section("RAW JSON OUTPUT", "─")
    print(json.dumps(report.model_dump(), indent=2))


# ---------------------------------------------------------------------------
# Streaming execution with step-level logging
# ---------------------------------------------------------------------------


async def _stream_pipeline(query: str) -> Optional[Dict[str, Any]]:
    """
    Execute the LangGraph pipeline in streaming mode, printing rich terminal
    output at each node transition.

    LangGraph's ``astream`` yields ``(node_name, partial_state_update)``
    tuples as each node completes, enabling real-time progress display.

    Parameters
    ----------
    query:
        The enterprise research prompt.

    Returns
    -------
    Optional[Dict[str, Any]]
        The final ``generated_report_draft`` dict, or ``None`` if the
        pipeline failed to produce any valid output.
    """
    init = initial_state(query)
    final_draft: Optional[Dict[str, Any]] = None
    pipeline_start = time.perf_counter()
    node_start: float = pipeline_start
    last_loop_counter: int = 0

    print(f"\n{_BOLD}Research Query:{_RESET}")
    print(f"  {query}\n")

    cfg = get_settings()
    print(f"{_BOLD}Configuration:{_RESET}")
    print(f"  Model          : {cfg.llm_model}")
    print(f"  Qdrant URL     : {cfg.qdrant_url}")
    print(f"  Collection     : {cfg.qdrant_collection}")
    print(f"  Circuit breaker: {cfg.circuit_breaker_max} max loops")
    print(f"  Max web results: {cfg.max_search_results}")
    print(f"  Max vector hits: {cfg.max_vector_results}")

    _print_section("PIPELINE EXECUTION LOG")

    async for event in app.astream(init, stream_mode="updates"):
        for node_name, state_update in event.items():
            elapsed = time.perf_counter() - node_start
            node_start = time.perf_counter()

            _print_node_start(node_name)

            # ---- ingest_data node ------------------------------------------
            if node_name == "ingest":
                ctx: list = state_update.get("retrieved_context", [])
                print(f"  Retrieved {len(ctx)} context chunks total.")
                _print_node_end(node_name, elapsed)

            # ---- generate_report node --------------------------------------
            elif node_name == "generate":
                draft = state_update.get("generated_report_draft")
                error = state_update.get("error_trace")
                if draft:
                    company = draft.get("company_name", "Unknown")
                    print(f"  ✓ Generated report for: {_BOLD}{company}{_RESET}")
                    final_draft = draft
                elif error:
                    preview = error[:200].replace("\n", " ")
                    print(f"  {_RED}✗ Schema validation failed.{_RESET}")
                    print(f"  Error preview: {preview}...")
                _print_node_end(node_name, elapsed)

            # ---- critic node -----------------------------------------------
            elif node_name == "critic":
                feedback: str = state_update.get("critic_feedback", "")
                loop_counter: int = state_update.get("loop_counter", 0)

                if feedback == "PASSED":
                    print(f"  {_GREEN}✓ Fact-check PASSED — all claims grounded.{_RESET}")
                else:
                    # Detect circuit breaker condition.
                    if loop_counter >= cfg.circuit_breaker_max:
                        _print_circuit_breaker(loop_counter)
                    else:
                        _print_loop_info(loop_counter, feedback)

                last_loop_counter = loop_counter
                _print_node_end(node_name, elapsed)

            else:
                # Unknown node — print raw update for debugging.
                print(f"  Update: {str(state_update)[:200]}")
                _print_node_end(node_name, elapsed)

    total_elapsed = time.perf_counter() - pipeline_start
    print(f"\n{_BOLD}Total pipeline elapsed time: {total_elapsed:.2f}s{_RESET}")
    print(f"{_BOLD}Total critic loops completed: {last_loop_counter}{_RESET}")

    return final_draft


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """
    Asynchronous main function: run the market intelligence pipeline end-to-end.

    Steps
    -----
    1. Print the application banner.
    2. Validate configuration (will raise on missing env vars).
    3. Stream the LangGraph pipeline with per-node terminal logging.
    4. Print the final validated JSON report.
    5. Exit with code 1 if no valid report was produced.
    """
    _print_banner()

    # Eagerly validate config — fail fast before touching any API.
    try:
        cfg = get_settings()
    except Exception as exc:
        print(
            f"\n{_RED}{_BOLD}Configuration error:{_RESET} {exc}\n"
            "Please ensure all required environment variables are set in your "
            ".env file or shell environment:\n"
            "  OPENAI_API_KEY, TAVILY_API_KEY\n"
        )
        sys.exit(1)

    query = DEMO_QUERY

    try:
        final_draft = await _stream_pipeline(query)
    except KeyboardInterrupt:
        print(f"\n{_YELLOW}Pipeline interrupted by user.{_RESET}")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unhandled exception during pipeline execution.")
        print(f"\n{_RED}{_BOLD}Fatal pipeline error:{_RESET} {exc}")
        sys.exit(1)

    if final_draft is None:
        print(
            f"\n{_RED}{_BOLD}ERROR:{_RESET} The pipeline completed without producing "
            "a valid structured report.\n"
            "Check the logs above for schema validation errors or API failures."
        )
        sys.exit(1)

    _print_final_report(final_draft)
    print(f"\n{_GREEN}{_BOLD}[✅ DONE]{_RESET}  Pipeline completed successfully.\n")


if __name__ == "__main__":
    asyncio.run(main())
