""""
schema.py
---------
All Pydantic models and TypedDicts that define the data contracts for the
Automated Enterprise Market Intelligence & Fact-Checking Engine.

Design principles
-----------------
* ``MarketIntelligenceReport`` is the canonical structured output that every
  other layer — generator, critic, and final consumer — reasons about.
* ``AgentState`` is the single mutable bag of state threaded through the
  entire LangGraph workflow.  Every node reads from it and returns a
  *partial* update dict; LangGraph merges those diffs automatically.
* ``CriticCorrection`` is a structured representation of a single field-level
  correction extracted from the critic's freeform feedback.  Parsing the
  critic's prose into typed objects lets the generator receive unambiguous,
  per-field mandates rather than a wall of text it can selectively ignore.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class MarketIntelligenceReport(BaseModel):
    """
    Canonical structured representation of a market intelligence analysis.

    All fields are required so that the LLM has no room to silently omit
    critical information.  The Critic agent validates every populated field
    against the retrieved ground-truth context.
    """

    company_name: str = Field(
        ...,
        min_length=1,
        description="Legal or commonly used name of the company or entity being analysed.",
    )
    market_cap_or_valuation: str = Field(
        ...,
        min_length=1,
        description=(
            "Current market capitalisation (public companies) or most recent "
            "private valuation (e.g. '$1.2T as of Q2 2025').  Must be a "
            "factual, source-backed figure — never a rough estimate."
        ),
    )
    core_revenue_drivers: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of the primary revenue segments or business drivers.  "
            "Each item must be a concrete, specific statement, not a generic label."
        ),
    )
    risk_factors: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "Material risks that could adversely impact the company's financial "
            "performance or market position.  Each item must be grounded in a "
            "specific retrieved source."
        ),
    )
    sources: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "URLs or descriptive citations for every factual claim made in this "
            "report.  At least one source is required."
        ),
    )

    @field_validator("core_revenue_drivers", "risk_factors", "sources", mode="before")
    @classmethod
    def non_empty_list(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("List field must contain at least one item.")
        return v

    @field_validator("company_name", "market_cap_or_valuation", mode="before")
    @classmethod
    def non_empty_string(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("String field must not be blank.")
        return v

    def to_display_dict(self) -> Dict[str, Any]:
        """Return a serialisable dict suitable for pretty-printing."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Structured critic correction — NEW
# ---------------------------------------------------------------------------


class CriticCorrection(BaseModel):
    """
    A single field-level correction extracted from the critic's freeform text.

    The critic writes prose like:
        "Field: market_cap_or_valuation
         Incorrect value: '$1.1 trillion as of December 2025'
         Context says: '$4.585 trillion to $4.6 trillion as of October 2025'
         Correct value: '$4.585–$4.6 trillion as of October 2025'"

    ``parse_critic_corrections`` extracts these into a list of
    ``CriticCorrection`` objects so the generator prompt can present per-field
    mandates instead of a block of prose the model can selectively ignore.

    Attributes
    ----------
    field_name
        One of the five ``MarketIntelligenceReport`` field names.
    rejected_value
        The verbatim value the generator produced that was rejected.
        Empty string when the critic did not quote it explicitly.
    correct_value
        The value the critic says should replace the rejected one.
        This is the most critical piece: it is injected directly into the
        revision prompt as a pre-filled answer the generator MUST use.
        Empty string when the critic only identified what was wrong but not
        what the right value is (rare; handled gracefully).
    evidence
        The context snippet or reasoning the critic cited as ground truth.
        Injected alongside ``correct_value`` so the generator can confirm
        the figure is traceable.
    """

    field_name: str
    rejected_value: str = ""
    correct_value: str = ""
    evidence: str = ""


# ---------------------------------------------------------------------------
# LangGraph agent state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    Mutable state bag threaded through every node of the LangGraph workflow.

    All fields are optional at the TypedDict level (``total=False``) because
    LangGraph merges partial update dicts — a node only needs to return the
    keys it actually modifies.

    Field semantics
    ---------------
    query
        The raw research question submitted by the end-user or CI pipeline.
    retrieved_context
        Aggregated text snippets from Qdrant (internal KB) and Tavily (live
        web).  Populated by ``ingest_data_node`` and read-only thereafter.
    generated_report_draft
        The most recent ``MarketIntelligenceReport`` serialised to a plain
        ``dict``.  ``None`` until the first successful generation pass.
    critic_feedback
        Raw text returned by the Critic agent.  Either the literal string
        ``'PASSED'`` (all facts grounded) or a detailed correction message.
    parsed_corrections
        Structured ``CriticCorrection`` objects extracted from
        ``critic_feedback`` by ``parse_critic_corrections``.  Used by
        ``generate_report_node`` to build per-field revision mandates.
        ``None`` on the first pass or when the critic passed.
    loop_counter
        Number of complete critic→generator correction cycles executed so far.
        Starts at ``0`` and is incremented by ``critic_node`` on each pass.
    error_trace
        String representation of the most recent exception raised during
        schema validation inside ``generate_report_node``.  ``None`` when the
        last generation attempt succeeded.  Cleared to ``None`` on success.
    previous_rejected_draft
        A copy of the last draft that the critic rejected, stored as a JSON
        string.  Used by ``generate_report_node`` on retry passes so the model
        can see exactly which output was rejected and revise it rather than
        regenerating from scratch.  ``None`` on the first generation pass.
    vagueness_error
        Set by ``generate_report_node`` when a revised field passes Pydantic
        schema validation but contains only generic/vague language instead of
        a concrete factual value.  Treated like ``error_trace`` — fed back
        into the next generation prompt as an explicit constraint.  ``None``
        when the last revision produced concrete content.
    """

    query: str
    retrieved_context: List[str]
    generated_report_draft: Optional[Dict[str, Any]]
    critic_feedback: Optional[str]
    parsed_corrections: Optional[List[Dict[str, str]]]   # ← NEW
    loop_counter: int
    error_trace: Optional[str]
    previous_rejected_draft: Optional[str]
    vagueness_error: Optional[str]                        # ← NEW


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def initial_state(query: str) -> AgentState:
    """
    Return a fully-initialised AgentState for a new research session.

    Using an explicit factory avoids accidental ``None``-key lookups in nodes
    that run before earlier nodes have populated all fields.
    """
    return AgentState(
        query=query,
        retrieved_context=[],
        generated_report_draft=None,
        critic_feedback=None,
        parsed_corrections=None,    # ← NEW
        loop_counter=0,
        error_trace=None,
        previous_rejected_draft=None,
        vagueness_error=None,       # ← NEW
    )
