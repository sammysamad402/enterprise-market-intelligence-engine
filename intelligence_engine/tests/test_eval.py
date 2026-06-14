"""
tests/test_eval.py
------------------
CI/CD evaluation harness for the Automated Enterprise Market Intelligence
& Fact-Checking Engine.

Evaluation framework
--------------------
* pytest         – test runner and assertion layer
* DeepEval       – LLM-as-judge metrics (Faithfulness, AnswerRelevancy)

The tests load the golden dataset from ``test_golden_dataset.json``, execute
the LangGraph pipeline asynchronously for each scenario, and assert that the
output satisfies quality thresholds suitable for a production intelligence
platform.

Running the tests
-----------------
From the repository root:

    pytest tests/test_eval.py -v --tb=short

Environment variables required
-------------------------------
    OPENAI_API_KEY      – used by the intelligence engine AND by DeepEval's judge model
    TAVILY_API_KEY      – used by the web search tool

    # Optional (override defaults)
    QDRANT_URL          – vector DB endpoint (default: http://localhost:6333)

Design notes
------------
* Each test case is parametrised so pytest produces individual pass/fail
  entries per scenario rather than a single monolithic test.
* ``asyncio.run()`` is used explicitly rather than the ``pytest-asyncio``
  plugin to keep the dependency surface small and avoid event-loop conflicts
  with LangGraph's internal async machinery.
* The score threshold is set at 0.50 for Faithfulness and 0.70 for
  AnswerRelevancy.  In a production CI pipeline you would tighten these
  thresholds (e.g. 0.85 / 0.90) once the system is stable and the golden
  dataset is curated.  They are intentionally relaxed here because the
  Faithfulness metric can score conservatively when the context snippets are
  mocked rather than live-retrieved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# ---------------------------------------------------------------------------
# Module setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("intelligence_engine.tests")

# Path to the golden dataset relative to this file.
GOLDEN_DATASET_PATH = Path(__file__).parent / "test_golden_dataset.json"

# Quality thresholds — raise these as the system matures.
FAITHFULNESS_THRESHOLD = 0.50
ANSWER_RELEVANCY_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_golden_dataset() -> List[Dict[str, Any]]:
    """Load and parse the golden dataset JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as fh:
        dataset: List[Dict[str, Any]] = json.load(fh)
    logger.info("Loaded %d golden test cases from %s", len(dataset), GOLDEN_DATASET_PATH)
    return dataset


def _report_dict_to_text(report_dict: Dict[str, Any] | None) -> str:
    """
    Convert a ``MarketIntelligenceReport`` dict to a readable plain-text
    string suitable for feeding into DeepEval's ``actual_output`` parameter.

    Parameters
    ----------
    report_dict:
        Serialised report dictionary, or ``None`` if generation failed.

    Returns
    -------
    str
        Human-readable text representation of the report.
    """
    if report_dict is None:
        return "ERROR: The pipeline failed to produce a valid report."

    lines: List[str] = [
        f"Company: {report_dict.get('company_name', 'N/A')}",
        f"Valuation: {report_dict.get('market_cap_or_valuation', 'N/A')}",
        "",
        "Core Revenue Drivers:",
    ]
    for driver in report_dict.get("core_revenue_drivers", []):
        lines.append(f"  - {driver}")

    lines.append("")
    lines.append("Risk Factors:")
    for risk in report_dict.get("risk_factors", []):
        lines.append(f"  - {risk}")

    lines.append("")
    lines.append("Sources:")
    for source in report_dict.get("sources", []):
        lines.append(f"  - {source}")

    return "\n".join(lines)


def _run_pipeline_for_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronously execute the LangGraph pipeline for a single test case.

    Uses ``asyncio.run`` to bridge the async pipeline into a synchronous
    pytest context.

    Parameters
    ----------
    test_case:
        A single entry from the golden dataset.

    Returns
    -------
    Dict[str, Any]
        The final ``AgentState`` after the pipeline has run to completion.
    """
    from intelligence_engine.graph import app
    from intelligence_engine.schema import initial_state

    query: str = test_case["input"]
    init_state = initial_state(query)

    async def _invoke() -> Dict[str, Any]:
        return await app.ainvoke(init_state)

    final_state: Dict[str, Any] = asyncio.run(_invoke())
    return final_state


# ---------------------------------------------------------------------------
# pytest parametrisation
# ---------------------------------------------------------------------------

_GOLDEN_DATASET: List[Dict[str, Any]] = _load_golden_dataset()


@pytest.mark.parametrize(
    "test_scenario",
    _GOLDEN_DATASET,
    ids=[case["id"] for case in _GOLDEN_DATASET],
)
def test_market_intelligence_faithfulness(test_scenario: Dict[str, Any]) -> None:
    """
    Assert that the generated report is faithful to the provided context.

    ``FaithfulnessMetric`` uses an LLM judge (powered by your ``OPENAI_API_KEY``)
    to check whether every factual claim in the ``actual_output`` can be
    supported by at least one of the ``retrieval_context`` snippets.

    A low faithfulness score indicates the generator is hallucinating facts
    not present in the retrieved ground truth.
    """
    logger.info(
        "Running faithfulness test for scenario: %s", test_scenario["id"]
    )

    # Execute the pipeline.
    final_state = _run_pipeline_for_case(test_scenario)
    report_text = _report_dict_to_text(final_state.get("generated_report_draft"))

    # Build the DeepEval test case.
    deepeval_case = LLMTestCase(
        input=test_scenario["input"],
        actual_output=report_text,
        retrieval_context=test_scenario["retrieval_context"],
    )

    # Instantiate and run the metric.
    metric = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model="gpt-4o",           # DeepEval judge model
        include_reason=True,      # surface reasoning in failure messages
    )
    metric.measure(deepeval_case)

    # Log full reasoning for debugging failed evaluations.
    logger.info(
        "Faithfulness score for %s: %.3f (threshold: %.2f)\nReason: %s",
        test_scenario["id"],
        metric.score,
        FAITHFULNESS_THRESHOLD,
        metric.reason,
    )

    assert metric.score >= FAITHFULNESS_THRESHOLD, (
        f"[{test_scenario['id']}] Faithfulness score {metric.score:.3f} "
        f"is below threshold {FAITHFULNESS_THRESHOLD:.2f}.\n"
        f"Reason: {metric.reason}\n\n"
        f"Generated output:\n{report_text}"
    )


@pytest.mark.parametrize(
    "test_scenario",
    _GOLDEN_DATASET,
    ids=[case["id"] for case in _GOLDEN_DATASET],
)
def test_market_intelligence_answer_relevancy(test_scenario: Dict[str, Any]) -> None:
    """
    Assert that the generated report is relevant to the original research query.

    ``AnswerRelevancyMetric`` uses an LLM judge to evaluate whether the
    actual output directly addresses the intent and content of the input
    query — i.e. the system is not going off-topic.
    """
    logger.info(
        "Running answer relevancy test for scenario: %s", test_scenario["id"]
    )

    # Execute the pipeline.
    final_state = _run_pipeline_for_case(test_scenario)
    report_text = _report_dict_to_text(final_state.get("generated_report_draft"))

    # Build the DeepEval test case.
    deepeval_case = LLMTestCase(
        input=test_scenario["input"],
        actual_output=report_text,
        retrieval_context=test_scenario["retrieval_context"],
    )

    # Instantiate and run the metric.
    metric = AnswerRelevancyMetric(
        threshold=ANSWER_RELEVANCY_THRESHOLD,
        model="gpt-4o",
        include_reason=True,
    )
    metric.measure(deepeval_case)

    logger.info(
        "Answer relevancy score for %s: %.3f (threshold: %.2f)\nReason: %s",
        test_scenario["id"],
        metric.score,
        ANSWER_RELEVANCY_THRESHOLD,
        metric.reason,
    )

    assert metric.score >= ANSWER_RELEVANCY_THRESHOLD, (
        f"[{test_scenario['id']}] Answer relevancy score {metric.score:.3f} "
        f"is below threshold {ANSWER_RELEVANCY_THRESHOLD:.2f}.\n"
        f"Reason: {metric.reason}\n\n"
        f"Generated output:\n{report_text}"
    )


@pytest.mark.parametrize(
    "test_scenario",
    _GOLDEN_DATASET,
    ids=[case["id"] for case in _GOLDEN_DATASET],
)
def test_market_intelligence_schema_validity(test_scenario: Dict[str, Any]) -> None:
    """
    Assert that the pipeline always produces a structurally valid report.

    This test does not use an LLM judge — it simply verifies that the
    generator produced a non-None draft with all required fields populated
    and that the Pydantic schema can be round-tripped cleanly.

    A failure here indicates the circuit breaker fired before a valid schema
    could be produced, which is a hard reliability regression.
    """
    from intelligence_engine.schema import MarketIntelligenceReport

    logger.info(
        "Running schema validity test for scenario: %s", test_scenario["id"]
    )

    final_state = _run_pipeline_for_case(test_scenario)
    draft = final_state.get("generated_report_draft")

    assert draft is not None, (
        f"[{test_scenario['id']}] Pipeline produced no valid draft. "
        f"error_trace: {final_state.get('error_trace')}"
    )

    # Validate by re-constructing the Pydantic model from the dict.
    try:
        report = MarketIntelligenceReport.model_validate(draft)
    except Exception as exc:
        pytest.fail(
            f"[{test_scenario['id']}] Draft failed Pydantic validation: {exc}\n"
            f"Draft content:\n{json.dumps(draft, indent=2)}"
        )

    # Field-level assertions.
    assert report.company_name.strip(), "company_name must not be blank"
    assert report.market_cap_or_valuation.strip(), "market_cap_or_valuation must not be blank"
    assert len(report.core_revenue_drivers) >= 1, "At least one revenue driver required"
    assert len(report.risk_factors) >= 1, "At least one risk factor required"
    assert len(report.sources) >= 1, "At least one source required"

    logger.info(
        "Schema validity check PASSED for %s — company=%r  sources=%d",
        test_scenario["id"],
        report.company_name,
        len(report.sources),
    )


# ---------------------------------------------------------------------------
# Batch evaluation entry point (optional — run standalone with DeepEval CLI)
# ---------------------------------------------------------------------------


def run_batch_evaluation() -> None:
    """
    Execute all three test scenarios through DeepEval's batch ``evaluate()``
    function.  This is useful for generating a DeepEval dashboard report
    rather than individual pytest assertions.

    Usage
    -----
        python -m tests.test_eval        # if run as a module
        deepeval test run tests/test_eval.py  # via DeepEval CLI
    """
    faithfulness_metric = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model="gpt-4o")
    relevancy_metric = AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD, model="gpt-4o")

    test_cases: List[LLMTestCase] = []

    for scenario in _GOLDEN_DATASET:
        final_state = _run_pipeline_for_case(scenario)
        report_text = _report_dict_to_text(final_state.get("generated_report_draft"))

        test_cases.append(
            LLMTestCase(
                input=scenario["input"],
                actual_output=report_text,
                retrieval_context=scenario["retrieval_context"],
            )
        )

    evaluate(
        test_cases=test_cases,
        metrics=[faithfulness_metric, relevancy_metric],
    )


if __name__ == "__main__":
    run_batch_evaluation()
