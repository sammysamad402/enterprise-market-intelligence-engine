"""
graph.py
--------
LangGraph workflow engine for the Automated Enterprise Market Intelligence
& Fact-Checking Engine.

Graph topology
--------------

    [START]
       │
       ▼
  ┌─────────────┐
  │ ingest_data │   ← concurrent Qdrant + Tavily retrieval via asyncio.gather
  └─────────────┘
       │
       ▼
  ┌───────────────────┐
  │ generate_report   │ ◄──────────────────────────────────┐
  └───────────────────┘                                    │
       │                                                   │
       ▼                                              (loop back)
  ┌────────────┐                                           │
  │   critic   │                                           │
  └────────────┘                                           │
       │                                                   │
       ▼                                                   │
  route_after_critic ──── loop_counter >= 3 ──────► [END] (circuit breaker)
       │                                                   │
       ├── error_trace set   ──────────────────────────────┤
       │                                                   │
       ├── critic != 'PASSED' ─────────────────────────────┘
       │
       └── PASSED ──────────────────────────────────► [END]

All nodes are async coroutines.  LangGraph supports async nodes natively
when the graph is compiled and invoked with ``await app.ainvoke(...)``.

Correction-loop design
----------------------
On loop_counter == 0  →  first-pass generation, no prior feedback.
On loop_counter >= 1  →  REVISION mode.  The model receives:
    * The previous rejected draft (verbatim JSON)
    * The critic's exact corrections
    * An explicit instruction to revise, not regenerate from scratch.

After each new draft is produced, ``generate_report_node`` compares it
field-by-field against the previous rejected draft and emits a WARNING log
for any field that the critic flagged but the generator left unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import traceback
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from intelligence_engine.config import get_llm, get_settings
from intelligence_engine.schema import (
    AgentState,
    CriticCorrection,
    MarketIntelligenceReport,
)
from intelligence_engine.tools import async_vector_query, async_web_query

logger = logging.getLogger("intelligence_engine.graph")


# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

# ── BEFORE (original) ────────────────────────────────────────────────────────
# The original _GENERATOR_SYSTEM_PROMPT injected critic_feedback as a
# freeform block labelled "PRIOR CRITIC FEEDBACK (if any)" with no
# structural distinction between a first-pass and a retry pass.  The model
# received the corrections as background context alongside the full context
# window and the research query, so the correction signal was diluted.
# Nothing in the prompt told the model *what to do differently*, nor was
# the previously rejected draft shown so the model could diff its own
# output.  The result: the model treated the correction as one noisy hint
# among many and often re-derived the same incorrect value from the raw
# context, especially when the context itself contained the wrong figure.
# ─────────────────────────────────────────────────────────────────────────────

# ── AFTER (new design) ───────────────────────────────────────────────────────
# The prompt is now *mode-aware*.  On loop_counter == 0 it is a clean first-
# pass generation prompt (no revision scaffolding shown).  On loop_counter >= 1
# it switches to an explicit REVISION prompt that:
#   1. Shows the rejected draft verbatim so the model can see exactly what it
#      produced.
#   2. Shows the critic corrections as a numbered checklist labelled MANDATORY.
#   3. Forbids the model from keeping any criticised claim unchanged.
#   4. Instructs the model to REVISE (targeted edits) rather than regenerate
#      from scratch (which risks forgetting valid fields).
# After generation, the node validates that corrected fields have actually
# changed value, and logs a WARNING if they have not.
# ─────────────────────────────────────────────────────────────────────────────

_GENERATOR_SYSTEM_PROMPT_FIRST_PASS = """\
You are a world-class enterprise market intelligence analyst.
Your task is to synthesise the provided research context into a structured,
fact-grounded market intelligence report for the company or entity specified
in the user query.

CRITICAL RULES
==============
1. You MUST respond with ONLY a single, valid JSON object that strictly
   conforms to the following schema.  Do NOT include any preamble, markdown
   fences, or explanatory text outside the JSON object.

2. JSON schema you must populate:
{{
  "company_name":           "<string — legal or common company name>",
  "market_cap_or_valuation":"<string — e.g. '$3.1T as of June 2025'>",
  "core_revenue_drivers":   ["<string>", ...],
  "risk_factors":           ["<string>", ...],
  "sources":                ["<URL or citation>", ...]
}}

3. Every factual claim MUST be traceable to the provided context snippets.
   Do NOT hallucinate figures, products, or statements absent from the context.

4. If the context lacks sufficient data for a field, write
   "Insufficient data in provided context" for that field — do NOT fabricate.

5. Populate "sources" with the URL or descriptive citation for each snippet
   you used.  At minimum one source is required.

PRIOR SCHEMA ERROR (if any)
=============================
{error_trace}

If the section above contains content, you MUST address those specific
schema errors before generating the JSON.
"""

_GENERATOR_SYSTEM_PROMPT_REVISION = """\
You are a world-class enterprise market intelligence analyst performing a
TARGETED REVISION of a previously rejected report.

A critic has reviewed your last output and identified specific factual errors.
Your ONLY task is to fix those errors.  Do NOT change any field that was NOT
criticised — preserve valid content verbatim.

CRITICAL RULES
==============
1. You MUST respond with ONLY a single, valid JSON object that strictly
   conforms to the following schema.  Do NOT include any preamble, markdown
   fences, or explanatory text outside the JSON object.

2. JSON schema you must populate:
{{
  "company_name":           "<string — legal or common company name>",
  "market_cap_or_valuation":"<string — e.g. '$3.1T as of June 2025'>",
  "core_revenue_drivers":   ["<string>", ...],
  "risk_factors":           ["<string>", ...],
  "sources":                ["<URL or citation>", ...]
}}

3. Every factual claim MUST be traceable to the provided context snippets.
   Do NOT hallucinate figures, products, or statements absent from the context.

4. If the context lacks sufficient data for a field, write
   "Insufficient data in provided context" for that field — do NOT fabricate.

5. Populate "sources" with the URL or descriptive citation for each snippet
   you used.  At minimum one source is required.

MANDATORY FIELD-LEVEL CORRECTIONS — YOU MUST APPLY EVERY ITEM
==============================================================
{structured_corrections_block}

ADDITIONAL CRITIC NOTES (supporting context — apply corrections above first)
=============================================================================
{raw_critic_feedback_block}

PRIOR SCHEMA ERROR (if any)
============================
{error_trace}

If the section above contains content, you MUST also address those specific
schema errors.

VAGUENESS ERROR FROM PREVIOUS REVISION (if any)
================================================
{vagueness_error}

If the vagueness section above is populated, the field it names MUST be
replaced with a concrete, source-backed value from the retrieved context —
do NOT repeat a generic placeholder.

REVISION INSTRUCTIONS
=====================
1. Start from the REJECTED DRAFT shown in the user message.
2. For every numbered correction above:
   a. Replace the field's current value with the MANDATORY VALUE given.
   b. If no mandatory value is provided, derive the correct answer from the
      retrieved context and ensure it is specific and source-backed.
3. Do NOT alter any field that was not named in the corrections.
4. Output the complete revised JSON (all five fields required).
5. Double-check: every corrected field MUST differ from the REJECTED DRAFT.
"""

_CRITIC_SYSTEM_PROMPT = """\
You are a meticulous fact-checking critic for a high-stakes enterprise
intelligence platform.

Your job is to cross-reference every claim in the DRAFT REPORT against the
GROUND-TRUTH CONTEXT snippets provided in the user message.

OUTPUT FORMAT
=============
Either respond with EXACTLY the string:
    PASSED
(if and ONLY IF every claim is directly supported by context), or respond with
one correction block per problematic field, using this EXACT structure:

    Field: <field_name>
    Incorrect claim: <verbatim value from the draft that is wrong>
    Context says: <the relevant quote or summary from the context snippets>
    Correct value: <the single definitive replacement value to use>

STRICT RULES
============
1. Use ONLY the field names: company_name, market_cap_or_valuation,
   core_revenue_drivers, risk_factors, sources.
   Do NOT mention a field name anywhere outside a "Field:" label line — prose
   that incidentally references a field name will confuse downstream parsing.

2. EVERY correction block MUST include a "Correct value:" line.
   - If the context contains the correct value, state it precisely and finally.
     Do NOT hedge with ranges when the context gives a specific figure.
   - If the context does NOT support the claim at all, write:
       Correct value: Remove unsupported claim
     This is the only acceptable substitute for a concrete value.
   - Never repeat a correction that asks for a minor wording tweak of a value
     you already specified as "Correct value" in a previous loop.

3. CONVERGENCE RULE — this is mandatory:
   If the draft's value for a field exactly matches a "Correct value" you
   (or a prior critic pass) already specified, you MUST accept it and NOT
   raise a new correction for that field — even if you now prefer slightly
   different wording.  Only raise a new correction if you can cite new
   evidence from the context that was not referenced before.

4. Do NOT be lenient about hallucinated figures.  Do NOT pass a report that
   contains a date or number absent from the context.

5. Do NOT give partial passes.  If one field is wrong, the whole report fails.
"""


# ---------------------------------------------------------------------------
# Helper: extract field names mentioned in critic feedback
# ---------------------------------------------------------------------------

# These are the five field names defined in MarketIntelligenceReport.
_ALL_REPORT_FIELDS = frozenset([
    "company_name",
    "market_cap_or_valuation",
    "core_revenue_drivers",
    "risk_factors",
    "sources",
])


def _fields_mentioned_in_feedback(feedback: str) -> set[str]:
    """
    Return the set of report field names that the critic *explicitly labelled*
    with a ``Field: <name>`` directive.

    This intentionally mirrors the strict-label-only logic in
    ``parse_critic_corrections``: a field name that appears incidentally in
    prose (e.g. "no sources support this") does NOT count as a criticised
    field, so ``_validate_corrections_applied`` will not emit a spurious
    "CORRECTION NOT APPLIED" warning for it.
    """
    explicitly_labelled: set[str] = set()
    for m in _FIELD_RE.finditer(feedback):
        candidate = m.group(1).strip()
        if candidate in _ALL_REPORT_FIELDS:
            explicitly_labelled.add(candidate)
    return explicitly_labelled


# ---------------------------------------------------------------------------
# Helper: parse critic prose into structured CriticCorrection objects
# ---------------------------------------------------------------------------

_FIELD_RE = re.compile(r"(?:Field|field)\s*[:\-]\s*([a-z_]+)", re.IGNORECASE)
_REJECTED_RE = re.compile(
    r"(?:Incorrect value|Incorrect claim|Rejected value|rejected|was)\s*[:\-]\s*['\"]?(.+?)['\"]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CORRECT_RE = re.compile(
    r"(?:Correct value|Should be|correct|replace with|use)\s*[:\-]\s*['\"]?(.+?)['\"]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONTEXT_RE = re.compile(
    r"(?:Context says|Evidence|Ground.truth|context)\s*[:\-]\s*['\"]?(.+?)['\"]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Sentinel phrases the critic may emit when context does not support a claim.
# Matched case-insensitively; the full set covers natural paraphrases.
_REMOVAL_SENTINELS: frozenset[str] = frozenset([
    "remove unsupported claim",
    "remove this claim",
    "remove claim",
    "delete unsupported claim",
    "omit unsupported claim",
    "unsupported — remove",
    "not supported — remove",
])

# Fields whose JSON type is List[str].  A removal instruction on these fields
# means "delete the criticised item from the list", NOT "replace the list with
# a string".  Fields absent from this set are scalar strings.
_LIST_TYPE_FIELDS: frozenset[str] = frozenset([
    "core_revenue_drivers",
    "risk_factors",
    "sources",
])


def parse_critic_corrections(feedback: str) -> list[CriticCorrection]:
    """
    Parse the critic's freeform correction text into a list of
    ``CriticCorrection`` objects — one per field-level issue found.

    The critic is prompted (via ``_CRITIC_SYSTEM_PROMPT``) to structure its
    output with labelled lines like::

        Field: market_cap_or_valuation
        Incorrect value: '$1.1 trillion as of December 2025'
        Context says: '$4.585 trillion to $4.6 trillion as of October 2025'
        Correct value: '$4.585–$4.6 trillion as of October 2025'

    **Strict-label-only parsing** — a ``CriticCorrection`` is emitted for a
    block if and only if an explicit ``Field: <name>`` label is present AND
    the extracted field name is one of the five valid report fields.  Incidental
    occurrences of a field name elsewhere in prose (e.g. "no sources support
    this claim") do NOT trigger a correction object.  This prevents
    ``_validate_corrections_applied`` from falsely flagging fields that the
    critic never actually criticised.

    Parameters
    ----------
    feedback:
        Raw string returned by ``critic_node`` (never ``'PASSED'``).

    Returns
    -------
    list[CriticCorrection]
        One entry per identified field-level problem.  May be empty if the
        critic's format is entirely unstructured and no field names can be
        found — the caller handles that gracefully by falling back to raw prose.
    """
    if not feedback or feedback.strip() == "PASSED":
        return []

    corrections: list[CriticCorrection] = []

    # Split on "Field:" markers to isolate per-field correction blocks.
    # Fallback to blank-line splitting only when no "Field:" labels exist.
    raw_blocks = re.split(r"\n(?=\s*[Ff]ield\s*[:\-])", feedback)
    if len(raw_blocks) == 1:
        raw_blocks = re.split(r"\n\s*\n", feedback)

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # ── STRICT: only accept blocks with an explicit "Field: <name>" label ──
        # The fallback that scanned for incidental field-name occurrences in
        # surrounding prose has been removed.  Without an explicit label we
        # cannot know which field the critic intended, so we skip the block.
        field_match = _FIELD_RE.search(block)
        if not field_match:
            logger.debug(
                "[parse_critic_corrections] Block has no explicit 'Field:' label — skipping.  "
                "Preview: %r",
                block[:120],
            )
            continue

        candidate = field_match.group(1).strip()
        if candidate not in _ALL_REPORT_FIELDS:
            logger.debug(
                "[parse_critic_corrections] Label %r is not a valid report field — skipping block.",
                candidate,
            )
            continue

        matched_field = candidate

        rejected_match = _REJECTED_RE.search(block)
        correct_match = _CORRECT_RE.search(block)
        context_match = _CONTEXT_RE.search(block)

        raw_correct = correct_match.group(1).strip() if correct_match else ""

        # Detect removal instructions.  When the critic writes
        # "Correct value: Remove unsupported claim" (or any recognised variant)
        # it is an instruction to delete the item, NOT a literal replacement
        # value.  We tag this so type-aware downstream code handles it correctly
        # for List[str] fields (item deletion) vs scalar fields (placeholder).
        is_removal = raw_correct.lower() in _REMOVAL_SENTINELS

        correction = CriticCorrection(
            field_name=matched_field,
            rejected_value=rejected_match.group(1).strip() if rejected_match else "",
            correct_value="" if is_removal else raw_correct,
            evidence=context_match.group(1).strip() if context_match else "",
            is_removal=is_removal,
        )
        corrections.append(correction)
        # INFO-level so every extracted correction is visible in the default log.
        logger.info(
            "[parse_critic_corrections] Extracted correction — "
            "field=%r  correct_value=%r  rejected_value=%r",
            correction.field_name,
            (correction.correct_value[:80] if correction.correct_value
             else "<none — derive from context>"),
            correction.rejected_value[:80] if correction.rejected_value else "<none>",
        )

    if not corrections:
        logger.warning(
            "[parse_critic_corrections] Could not parse any structured "
            "corrections from critic feedback — falling back to raw prose.  "
            "Feedback preview: %s",
            feedback[:200],
        )

    return corrections


def _parse_and_log_corrections(feedback: str) -> list[dict]:
    """
    Run ``parse_critic_corrections`` and serialise to list[dict] for storage
    in ``AgentState`` (TypedDict requires plain dicts, not Pydantic models).

    Logs a structured summary at INFO level so the field names and mandatory
    replacement values are always visible without enabling DEBUG logging.
    """
    corrections = parse_critic_corrections(feedback)
    field_summary = [
        f"{c.field_name!r} → {c.correct_value[:60]!r}" if c.correct_value
        else f"{c.field_name!r} → <derive from context>"
        for c in corrections
    ]
    logger.info(
        "[critic_node] Parsed %d structured correction(s) from feedback.  "
        "Fields & mandatory values: [%s]",
        len(corrections),
        " | ".join(field_summary) if field_summary else "none",
    )
    return [c.model_dump() for c in corrections]


# ---------------------------------------------------------------------------
# Node: ingest_data_node
# ---------------------------------------------------------------------------


async def ingest_data_node(state: AgentState) -> Dict[str, Any]:
    """
    Concurrently fetch context from Qdrant (internal KB) and Tavily (live web).

    Both I/O operations are fired simultaneously with ``asyncio.gather`` so
    the total latency is ``max(qdrant_latency, tavily_latency)`` rather than
    the sum of both.

    Parameters
    ----------
    state:
        Current agent state.  Only ``state["query"]`` is read.

    Returns
    -------
    Dict[str, Any]
        Partial state update containing ``retrieved_context``.
    """
    query: str = state["query"]
    logger.info("[ingest_data_node] Starting concurrent retrieval — query=%r", query[:80])

    vector_results, web_results = await asyncio.gather(
        async_vector_query(query),
        async_web_query(query),
        return_exceptions=False,  # individual tools handle their own exceptions
    )

    # Merge and deduplicate while preserving order (vector results first so
    # internal KB has higher priority in the context window).
    seen: set[str] = set()
    combined: list[str] = []
    for snippet in (*vector_results, *web_results):
        if snippet not in seen:
            seen.add(snippet)
            combined.append(snippet)

    logger.info(
        "[ingest_data_node] Retrieval complete — "
        "vector_hits=%d  web_hits=%d  combined=%d",
        len(vector_results),
        len(web_results),
        len(combined),
    )

    return {"retrieved_context": combined}


# ---------------------------------------------------------------------------
# Helpers: structured correction rendering + vagueness detection
# ---------------------------------------------------------------------------

_VAGUE_SENTINELS = frozenset([
    "insufficient data",
    "not available",
    "n/a",
    "unknown",
    "tbd",
    "to be determined",
    "no data",
    "not specified",
    "not provided",
    "see context",
    "refer to context",
])

# Fields on which vagueness detection is enforced.
_VAGUE_FIELDS = frozenset([
    "market_cap_or_valuation",
])


def _render_structured_corrections(
    corrections: list[CriticCorrection],
) -> str:
    """
    Render a list of ``CriticCorrection`` objects into a numbered checklist
    string for injection into ``_GENERATOR_SYSTEM_PROMPT_REVISION``.

    Type-aware rendering
    --------------------
    For **list-type fields** (``core_revenue_drivers``, ``risk_factors``,
    ``sources``) where the critic issued a removal instruction:
      → The model is told to *remove the specific item* from the list and
        keep all other items intact.  It is explicitly forbidden from
        replacing the entire list with a string.

    For **scalar fields** where the critic issued a removal instruction:
      → The model is told to replace the value with
        ``"Insufficient data in provided context"``.

    For normal value-replacement corrections (no removal):
      → Existing MANDATORY VALUE behaviour is preserved unchanged.
    """
    if not corrections:
        return "No structured corrections available — refer to the critic notes below."

    lines: list[str] = []
    for i, c in enumerate(corrections, start=1):
        parts = [f"{i}. FIELD: {c.field_name}"]
        if c.rejected_value:
            parts.append(f"   REJECTED VALUE  : {c.rejected_value}")

        if c.is_removal:
            if c.field_name in _LIST_TYPE_FIELDS:
                # List field — delete only the offending item.
                item_hint = (
                    f" (the item matching: {c.rejected_value!r})"
                    if c.rejected_value
                    else ""
                )
                parts.append(
                    f"   INSTRUCTION     : REMOVE the specific list item{item_hint} "
                    f"from {c.field_name}. "
                    f"Keep ALL other list items exactly as-is. "
                    f"The field MUST remain a JSON array — do NOT replace the "
                    f"entire array with a string value."
                )
            else:
                # Scalar field — replace with a safe placeholder.
                parts.append(
                    f"   INSTRUCTION     : REMOVE the unsupported value from "
                    f"{c.field_name}. Replace it with "
                    f'"Insufficient data in provided context".'
                )
        elif c.correct_value:
            parts.append(
                f"   MANDATORY VALUE : {c.correct_value}  \u2190 USE THIS EXACTLY"
            )
        else:
            parts.append(
                "   MANDATORY VALUE : (derive from context — must be concrete and specific)"
            )

        if c.evidence:
            parts.append(f"   EVIDENCE        : {c.evidence}")
        lines.append("\n".join(parts))

    return "\n\n".join(lines)


def _detect_vagueness(report: MarketIntelligenceReport) -> str | None:
    """
    Return a human-readable error string if any scalar field that should carry
    a concrete value contains only a generic placeholder.  Return ``None`` when
    the report looks sufficiently specific.

    Only fields listed in ``_VAGUE_FIELDS`` are checked.  Extend that set to
    enforce vagueness detection on additional fields.
    """
    for field_name in _VAGUE_FIELDS:
        value: str = getattr(report, field_name, "") or ""
        if any(sentinel in value.lower() for sentinel in _VAGUE_SENTINELS):
            return (
                f"Field '{field_name}' contains a vague placeholder: {value!r}.  "
                "You MUST replace it with a specific, source-backed value from the "
                "retrieved context.  Do NOT use generic phrases like 'not available' "
                "or 'insufficient data' — find the actual figure in the context."
            )
    return None


def _apply_removal_corrections(
    raw_dict: dict,
    parsed_corrections: list[CriticCorrection],
    previous_draft: dict | None,
) -> dict:
    """
    Sanitise a raw JSON dict produced by the LLM before Pydantic validation.

    This is the last line of defence against the "Remove unsupported claim"
    schema corruption bug.  Even with improved prompt language, an LLM may
    still literally set a List[str] field to the string
    ``"Remove unsupported claim"``.  This function detects and repairs that.

    Repair strategy per field type
    --------------------------------
    **List[str] fields** (``core_revenue_drivers``, ``risk_factors``, ``sources``):

    Case A — LLM produced a string instead of a list (the core bug):
        The field value is a string (e.g. ``"Remove unsupported claim"``).
        We fall back to the *previous draft's* list for that field and apply
        the item deletion there instead, because the LLM's output is
        structurally invalid and cannot be trusted.

    Case B — LLM produced a list but left a removal-sentinel string as an
        element (e.g. ``["valid item", "Remove unsupported claim"]``):
        Remove any list items that match a removal sentinel and any items
        that exactly match a ``rejected_value`` from a removal correction.

    **Scalar fields** (``company_name``, ``market_cap_or_valuation``):
        If the value matches a removal sentinel, replace it with the safe
        placeholder ``"Insufficient data in provided context"``.

    Parameters
    ----------
    raw_dict:
        The parsed JSON dict returned by the LLM (may be malformed).
    parsed_corrections:
        Structured corrections from the last critic pass.
    previous_draft:
        The last accepted (or best-effort) draft dict.  Used as a fallback
        for list fields when the LLM has replaced the list with a string.

    Returns
    -------
    dict
        A sanitised copy of ``raw_dict`` safe to pass to Pydantic.
    """
    if not parsed_corrections:
        return raw_dict

    result = dict(raw_dict)  # shallow copy — we only mutate top-level fields

    # Build lookup: field_name → list of removal corrections for that field.
    removals: dict[str, list[CriticCorrection]] = {}
    for c in parsed_corrections:
        if c.is_removal:
            removals.setdefault(c.field_name, []).append(c)

    if not removals:
        return result  # no removal corrections — nothing to do

    for field_name, corrections in removals.items():
        current_value = result.get(field_name)

        if field_name in _LIST_TYPE_FIELDS:
            # ── Case A: LLM replaced the list with a string ──────────────
            if not isinstance(current_value, list):
                logger.warning(
                    "[_apply_removal_corrections] Field %r is %s instead of a list "
                    "— LLM appears to have applied the removal sentinel literally.  "
                    "Falling back to previous draft and removing the flagged items.",
                    field_name,
                    type(current_value).__name__,
                )
                # Use the previous draft's list as the base for deletion.
                base_list: list[str] = []
                if previous_draft and isinstance(previous_draft.get(field_name), list):
                    base_list = list(previous_draft[field_name])
                else:
                    logger.warning(
                        "[_apply_removal_corrections] No previous draft available "
                        "for field %r — field will be set to empty list.",
                        field_name,
                    )
                current_value = base_list

            # ── Case B (and Case A after fallback): filter out removed items ──
            rejected_items: set[str] = {
                c.rejected_value.strip().strip('"\'')
                for c in corrections
                if c.rejected_value
            }

            def _should_remove(item: Any) -> bool:
                """True if item is a removal sentinel or matches a rejected value."""
                if not isinstance(item, str):
                    return False
                item_lower = item.strip().lower()
                if item_lower in _REMOVAL_SENTINELS:
                    return True
                # Fuzzy match: item contains or is contained by a rejected value.
                for rv in rejected_items:
                    if rv and (rv.lower() in item_lower or item_lower in rv.lower()):
                        return True
                return False

            filtered = [item for item in current_value if not _should_remove(item)]

            removed_count = len(current_value) - len(filtered)
            if removed_count:
                logger.info(
                    "[_apply_removal_corrections] Removed %d item(s) from list "
                    "field %r.  Remaining items: %d.",
                    removed_count,
                    field_name,
                    len(filtered),
                )
            else:
                logger.debug(
                    "[_apply_removal_corrections] No items matched removal criteria "
                    "for field %r.",
                    field_name,
                )

            # Ensure at least one item remains so Pydantic's min_length=1 passes.
            # If we emptied the list the critic will catch it next loop and the
            # generator will be asked to populate it from context.
            result[field_name] = filtered if filtered else ["Insufficient data in provided context"]

        else:
            # ── Scalar field ──────────────────────────────────────────────
            if isinstance(current_value, str) and (
                current_value.strip().lower() in _REMOVAL_SENTINELS
                or not current_value.strip()
            ):
                logger.info(
                    "[_apply_removal_corrections] Scalar field %r contained a "
                    "removal sentinel — replacing with safe placeholder.",
                    field_name,
                )
                result[field_name] = "Insufficient data in provided context"

    return result


# ---------------------------------------------------------------------------
# Node: generate_report_node
# ---------------------------------------------------------------------------


async def generate_report_node(state: AgentState) -> Dict[str, Any]:
    """
    Invoke the LLM to generate or revise a structured ``MarketIntelligenceReport``.

    Mode selection
    --------------
    * ``loop_counter == 0`` → **First-pass generation.**
      Clean prompt; no revision scaffolding.

    * ``loop_counter >= 1`` → **Revision mode.**
      The prompt now uses *structured* per-field corrections extracted from
      ``state["parsed_corrections"]`` rather than the raw critic prose, so the
      model receives unambiguous, mandatory replacements (MANDATORY VALUE labels)
      rather than a block of text it can selectively interpret.  The raw critic
      prose is still shown as a supplementary note so nuance is not lost.

    Post-generation validation
    --------------------------
    1. Pydantic schema validation (unchanged).
    2. Vagueness detection — if a scalar field like ``market_cap_or_valuation``
       still contains a generic placeholder after revision, ``vagueness_error``
       is written to state so the next prompt names the exact field and demands
       a concrete, source-backed value.
    3. Field-change diff against the rejected draft (unchanged).

    Parameters
    ----------
    state:
        Current agent state.

    Returns
    -------
    Dict[str, Any]
        Partial state update with ``generated_report_draft``, ``error_trace``,
        ``previous_rejected_draft``, and ``vagueness_error``.
    """
    query: str = state["query"]
    context_snippets: list[str] = state.get("retrieved_context", [])
    critic_feedback: str = state.get("critic_feedback") or "None"
    error_trace: str = state.get("error_trace") or "None"
    vagueness_error: str = state.get("vagueness_error") or "None"
    loop_counter: int = state.get("loop_counter", 0)
    previous_rejected_draft: str | None = state.get("previous_rejected_draft")
    parsed_corrections_raw: list[dict] | None = state.get("parsed_corrections")

    is_revision = loop_counter > 0

    # ------------------------------------------------------------------
    # Deserialise parsed_corrections from state (stored as list[dict]).
    # ------------------------------------------------------------------
    parsed_corrections: list[CriticCorrection] = []
    if parsed_corrections_raw:
        try:
            parsed_corrections = [
                CriticCorrection(**c) for c in parsed_corrections_raw
            ]
        except Exception as exc:
            logger.warning(
                "[generate_report_node] Could not deserialise parsed_corrections "
                "from state — %s.  Falling back to raw critic prose.",
                exc,
            )

    logger.info(
        "[generate_report_node] %s — loop=%d  context_chunks=%d  "
        "parsed_corrections=%d",
        "REVISION" if is_revision else "First-pass generation",
        loop_counter,
        len(context_snippets),
        len(parsed_corrections),
    )

    # ------------------------------------------------------------------
    # Build the context block injected into the user message.
    # ------------------------------------------------------------------
    if context_snippets:
        context_block = "\n\n".join(
            f"[SNIPPET {i + 1}]\n{snippet}"
            for i, snippet in enumerate(context_snippets)
        )
    else:
        context_block = (
            "NO CONTEXT RETRIEVED — base your report solely on well-established "
            "public knowledge and clearly mark all fields as potentially unverified."
        )

    # ------------------------------------------------------------------
    # Select prompt template based on whether this is a revision pass.
    # ------------------------------------------------------------------
    if is_revision:
        # Render structured corrections into a numbered checklist with
        # MANDATORY VALUE labels derived from CriticCorrection.correct_value.
        structured_block = _render_structured_corrections(parsed_corrections)

        # Also supply the raw prose as a supplementary note so the model has
        # access to any reasoning the structured parser may have missed.
        raw_block = (
            critic_feedback
            if critic_feedback and critic_feedback != "None"
            else "None"
        )

        system_prompt = _GENERATOR_SYSTEM_PROMPT_REVISION.format(
            structured_corrections_block=structured_block,
            raw_critic_feedback_block=raw_block,
            error_trace=error_trace,
            vagueness_error=vagueness_error,
        )

        rejected_draft_block = (
            previous_rejected_draft
            if previous_rejected_draft
            else "No previous draft available."
        )
        user_message = (
            f"RESEARCH QUERY: {query}\n\n"
            f"RETRIEVED CONTEXT\n{'=' * 60}\n{context_block}\n{'=' * 60}\n\n"
            f"REJECTED DRAFT (your previous output — DO NOT copy unchanged fields "
            f"that were criticised)\n{'=' * 60}\n{rejected_draft_block}\n{'=' * 60}\n\n"
            "Produce the revised market intelligence report JSON now.\n"
            "Remember: every field named in the corrections checklist MUST have a "
            "different value from the REJECTED DRAFT above."
        )
    else:
        system_prompt = _GENERATOR_SYSTEM_PROMPT_FIRST_PASS.format(
            error_trace=error_trace,
        )
        user_message = (
            f"RESEARCH QUERY: {query}\n\n"
            f"RETRIEVED CONTEXT\n{'=' * 60}\n{context_block}\n{'=' * 60}\n\n"
            "Generate the market intelligence report JSON now."
        )

    llm = get_llm()

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        )

        raw_output: str = response.content  # type: ignore[assignment]
        logger.debug("[generate_report_node] Raw LLM output:\n%s", raw_output[:500])

        # ------------------------------------------------------------------
        # Strip optional markdown code fences that some model versions emit
        # despite instructions.
        # ------------------------------------------------------------------
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            inner_lines = lines[1:] if lines[-1].strip() == "```" else lines[1:]
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            cleaned = "\n".join(inner_lines).strip()

        # ------------------------------------------------------------------
        # Parse to raw dict first so we can sanitise removal instructions
        # before Pydantic validation.  This is the layer that catches the
        # "risk_factors": "Remove unsupported claim" class of bugs.
        # ------------------------------------------------------------------
        try:
            raw_dict: dict = json.loads(cleaned)
        except json.JSONDecodeError as json_exc:
            raise ValueError(
                f"LLM output is not valid JSON: {json_exc}\n\nRaw output:\n{cleaned[:500]}"
            ) from json_exc

        previous_draft_dict: dict | None = None
        if previous_rejected_draft:
            try:
                previous_draft_dict = json.loads(previous_rejected_draft)
            except (json.JSONDecodeError, TypeError):
                pass

        sanitised_dict = _apply_removal_corrections(
            raw_dict=raw_dict,
            parsed_corrections=parsed_corrections,
            previous_draft=previous_draft_dict,
        )

        # ------------------------------------------------------------------
        # Pydantic validation on the sanitised dict.
        # ------------------------------------------------------------------
        report = MarketIntelligenceReport.model_validate(sanitised_dict)

        logger.info(
            "[generate_report_node] Schema validation PASSED — company=%r",
            report.company_name,
        )

        new_draft = report.model_dump()

        # ------------------------------------------------------------------
        # Vagueness detection — NEW.
        # Check scalar fields for generic placeholder language.
        # ------------------------------------------------------------------
        vagueness_msg: str | None = _detect_vagueness(report)
        if vagueness_msg:
            logger.warning(
                "[generate_report_node] ⚠️  VAGUENESS DETECTED — %s",
                vagueness_msg,
            )

        # ------------------------------------------------------------------
        # Post-generation correction validation (loop_counter > 0 only).
        # Compare criticised fields against the previous rejected draft to
        # detect if the model silently kept a value it was told to change.
        # ------------------------------------------------------------------
        if is_revision and previous_rejected_draft and critic_feedback != "None":
            _validate_corrections_applied(
                new_draft=new_draft,
                rejected_draft_json=previous_rejected_draft,
                critic_feedback=critic_feedback,
            )

        return {
            "generated_report_draft": new_draft,
            "error_trace": None,                    # clear any previous error
            "previous_rejected_draft": None,        # reset; critic will set it if needed
            "vagueness_error": vagueness_msg,       # None clears it when clean
        }

    except Exception as exc:  # noqa: BLE001
        tb_str = traceback.format_exc()
        logger.error(
            "[generate_report_node] Schema validation FAILED — %s: %s",
            type(exc).__name__,
            exc,
        )
        logger.debug("[generate_report_node] Full traceback:\n%s", tb_str)

        return {
            "generated_report_draft": None,
            "error_trace": (
                f"Exception type : {type(exc).__name__}\n"
                f"Exception value: {exc}\n\n"
                f"Full traceback:\n{tb_str}"
            ),
            "previous_rejected_draft": previous_rejected_draft,  # preserve for retry
            "vagueness_error": None,                # schema failed; don't carry stale msg
        }


# ---------------------------------------------------------------------------
# Helper: validate that critic corrections were actually applied
# ---------------------------------------------------------------------------


def _validate_corrections_applied(
    new_draft: dict,
    rejected_draft_json: str,
    critic_feedback: str,
) -> None:
    """
    Compare ``new_draft`` against the previously rejected draft for every field
    that appears in the critic feedback.  Log a WARNING for any field whose
    value is unchanged, because that indicates the model ignored the correction.

    This does not raise or block — it is a diagnostic / observability tool.
    The correction-loop itself will catch residual errors on the next critic pass.
    """
    try:
        old_draft: dict = json.loads(rejected_draft_json)
    except (json.JSONDecodeError, TypeError):
        logger.debug(
            "[_validate_corrections_applied] Could not parse previous draft JSON — skipping comparison."
        )
        return

    criticised_fields = _fields_mentioned_in_feedback(critic_feedback)
    if not criticised_fields:
        logger.debug(
            "[_validate_corrections_applied] No recognisable field names in critic feedback — skipping."
        )
        return

    for field in criticised_fields:
        old_val = old_draft.get(field)
        new_val = new_draft.get(field)

        if old_val == new_val:
            logger.warning(
                "[generate_report_node] ⚠️  CORRECTION NOT APPLIED — "
                "field=%r was criticised but its value is UNCHANGED.\n"
                "  Rejected value : %r\n"
                "  New value      : %r\n"
                "  Critic feedback: %s",
                field,
                old_val,
                new_val,
                critic_feedback[:300],
            )
        else:
            logger.info(
                "[generate_report_node] ✓ Correction applied — field=%r changed.",
                field,
            )


# ---------------------------------------------------------------------------
# Node: critic_node
# ---------------------------------------------------------------------------


async def critic_node(state: AgentState) -> Dict[str, Any]:
    """
    Prompt the LLM to aggressively fact-check the generated report draft.

    The Critic receives:
    * The original research query (for relevance checking)
    * All retrieved context snippets (ground truth)
    * The generated report draft (to be verified)

    It returns either the literal string ``'PASSED'`` or a detailed correction
    message.  The ``loop_counter`` is incremented regardless of outcome.

    When the draft is rejected, the current draft is serialised into
    ``previous_rejected_draft`` so that the next generation pass can display
    it to the model verbatim and ask for a targeted revision.

    On rejection, ``parse_critic_corrections`` is called to extract structured
    ``CriticCorrection`` objects from the freeform feedback text.  These are
    stored in ``parsed_corrections`` (as ``list[dict]``) so that
    ``generate_report_node`` can build a numbered MANDATORY VALUE checklist
    instead of relying on the raw prose.

    Parameters
    ----------
    state:
        Current agent state.

    Returns
    -------
    Dict[str, Any]
        Partial state update with ``critic_feedback``, ``loop_counter``,
        (when rejecting) ``previous_rejected_draft``, and ``parsed_corrections``.
    """
    draft: dict | None = state.get("generated_report_draft")
    context_snippets: list[str] = state.get("retrieved_context", [])
    loop_counter: int = state.get("loop_counter", 0)
    new_loop_counter = loop_counter + 1

    logger.info(
        "[critic_node] Running fact-check — loop=%d  draft_present=%s",
        new_loop_counter,
        draft is not None,
    )

    # If the generator failed to produce a valid draft, short-circuit and
    # ask for a retry without wasting a critic LLM call.
    if draft is None:
        logger.warning(
            "[critic_node] No valid draft to review — "
            "returning NEEDS_SCHEMA_FIX feedback."
        )
        return {
            "critic_feedback": (
                "NEEDS_SCHEMA_FIX: The previous generation attempt did not produce "
                "a valid JSON report.  Please correct the schema errors detailed in "
                "error_trace and try again."
            ),
            "loop_counter": new_loop_counter,
            # previous_rejected_draft stays as-is (no new draft to save)
        }

    # ------------------------------------------------------------------
    # Build context block.
    # ------------------------------------------------------------------
    if context_snippets:
        context_block = "\n\n".join(
            f"[SNIPPET {i + 1}]\n{snippet}"
            for i, snippet in enumerate(context_snippets)
        )
    else:
        context_block = "NO CONTEXT AVAILABLE — all claims are potentially ungrounded."

    draft_json = json.dumps(draft, indent=2)

    # Build a "prior mandates" block so the critic can apply its own
    # convergence rule: if the draft now matches a value the critic already
    # specified as "Correct value", it must accept it.
    prior_corrections_raw: list[dict] | None = state.get("parsed_corrections")
    prior_mandates_block = ""
    if prior_corrections_raw:
        mandate_lines = [
            f"  - Field: {c['field_name']}  →  Correct value previously mandated: "
            f"{c['correct_value'] or '<derive from context>'}"
            for c in prior_corrections_raw
            if c.get("field_name")
        ]
        if mandate_lines:
            prior_mandates_block = (
                "\nPREVIOUSLY MANDATED CORRECTIONS (apply convergence rule)\n"
                + "=" * 60 + "\n"
                + "If the draft now matches any of these values, do NOT raise a new\n"
                  "correction for that field unless you cite new context evidence.\n"
                + "\n".join(mandate_lines)
                + "\n" + "=" * 60 + "\n"
            )

    user_message = (
        f"ORIGINAL QUERY: {state.get('query', '')}\n\n"
        f"GROUND-TRUTH CONTEXT\n{'=' * 60}\n{context_block}\n{'=' * 60}\n"
        f"{prior_mandates_block}\n"
        f"DRAFT REPORT TO VERIFY\n{'=' * 60}\n{draft_json}\n{'=' * 60}\n\n"
        "Begin your fact-check now.  Output ONLY 'PASSED' or your detailed "
        "correction message."
    )

    llm = get_llm()

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=_CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
        feedback: str = response.content.strip()  # type: ignore[union-attr]

        if feedback == "PASSED":
            logger.info("[critic_node] Fact-check result: PASSED ✓")
            return {
                "critic_feedback": feedback,
                "loop_counter": new_loop_counter,
                "parsed_corrections": [],   # clear stale corrections on PASS
                # Do not set previous_rejected_draft — it's not needed on PASS.
            }
        else:
            logger.warning(
                "[critic_node] Fact-check result: FAILED — corrections requested.\n%s",
                feedback[:300],
            )
            # Parse structured corrections for generate_report_node.
            parsed = _parse_and_log_corrections(feedback)
            # Serialise the rejected draft now so generate_report_node can show it
            # verbatim to the model on the next revision pass.
            return {
                "critic_feedback": feedback,
                "loop_counter": new_loop_counter,
                "previous_rejected_draft": draft_json,
                "parsed_corrections": parsed,
            }

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[critic_node] LLM invocation failed — %s: %s  "
            "Treating as a failure to force a regeneration attempt.",
            type(exc).__name__,
            exc,
        )
        parsed = _parse_and_log_corrections(
            f"CRITIC_ERROR: {type(exc).__name__}: {exc}"
        )
        return {
            "critic_feedback": (
                f"CRITIC_ERROR: The critic agent raised an exception: "
                f"{type(exc).__name__}: {exc}"
            ),
            "loop_counter": new_loop_counter,
            "previous_rejected_draft": draft_json,
            "parsed_corrections": parsed,
        }


# ---------------------------------------------------------------------------
# Conditional router: route_after_critic
# ---------------------------------------------------------------------------


def route_after_critic(state: AgentState) -> str:
    """
    Determine the next graph node after the Critic has run.

    Routing logic (evaluated in priority order)
    -------------------------------------------
    1. ``loop_counter >= circuit_breaker_max``
       → Circuit breaker: log a warning and route to ``END`` to prevent
         infinite token-drain loops.

    2. ``error_trace is not None``
       → The generator failed schema validation on the last pass.
         Route back to ``"generate"`` with the error context so it can fix
         the malformed output.

    3. ``critic_feedback != 'PASSED'``
       → The Critic found ungrounded claims.  Route back to ``"generate"``
         with the correction feedback.

    4. Default
       → ``critic_feedback == 'PASSED'``.  Route to ``END``; the report is
         accepted as valid and grounded.

    Parameters
    ----------
    state:
        Current agent state after ``critic_node`` has updated it.

    Returns
    -------
    str
        Either ``"generate"`` or the LangGraph ``END`` sentinel.
    """
    cfg = get_settings()
    loop_counter: int = state.get("loop_counter", 0)
    error_trace: str | None = state.get("error_trace")
    critic_feedback: str | None = state.get("critic_feedback")

    # ---- 1. Circuit breaker -------------------------------------------------
    if loop_counter >= cfg.circuit_breaker_max:
        logger.warning(
            "[route_after_critic] ⚡ CIRCUIT BREAKER TRIGGERED — "
            "loop_counter=%d >= max=%d.  "
            "Routing to END with the current best-effort draft.",
            loop_counter,
            cfg.circuit_breaker_max,
        )
        return END  # type: ignore[return-value]

    # ---- 2. Schema error needs fixing ---------------------------------------
    if error_trace is not None:
        logger.info(
            "[route_after_critic] Schema error detected — routing back to generate."
        )
        return "generate"

    # ---- 3. Critic found factual issues ------------------------------------
    if critic_feedback != "PASSED":
        logger.info(
            "[route_after_critic] Critic rejected the draft — "
            "routing back to generate with corrections."
        )
        return "generate"

    # ---- 4. Report accepted -------------------------------------------------
    logger.info(
        "[route_after_critic] ✅ Report ACCEPTED by critic — routing to END."
    )
    return END  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """
    Assemble and compile the LangGraph ``StateGraph``.

    Returns
    -------
    CompiledGraph
        An executable LangGraph application object.
    """
    workflow = StateGraph(AgentState)

    # -- Register nodes -------------------------------------------------------
    workflow.add_node("ingest", ingest_data_node)
    workflow.add_node("generate", generate_report_node)
    workflow.add_node("critic", critic_node)

    # -- Define edges ---------------------------------------------------------
    workflow.add_edge(START, "ingest")
    workflow.add_edge("ingest", "generate")
    workflow.add_edge("generate", "critic")

    # -- Conditional router after critic --------------------------------------
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "generate": "generate",  # loop back for corrections
            END: END,                # accept or circuit-break
        },
    )

    compiled = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Module-level compiled application (imported by main.py and test_eval.py)
# ---------------------------------------------------------------------------

app = build_graph()