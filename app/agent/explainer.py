"""Advisory explainer: turns engine results into a plain-language summary.

The model only narrates. All figures are formatted in Python and passed in; the reply is
checked so every number it contains traces back to those inputs. If the model introduces
a number of its own twice in a row, a deterministic summary is returned instead.
"""

from __future__ import annotations

import re
from typing import Any

from google import genai
from google.genai import types

from app.agent.client import generate_content, get_client
from app.models.profile import UserProfileSchema

DISCLAIMER = (
    "Simulated projections based on IALM 2012-14 actuarial benchmarks and standard banking FOIR "
    "guidelines. This platform is an educational simulation tool, not an authorized SEBI or IRDAI "
    "financial advisor."
)

SYSTEM_INSTRUCTION = """You write an executive summary of a financial plan for an Indian household.
Tone: analytical, objective and neutral. Never salesy or promotional; no exclamation marks;
do not recommend specific products, brands or insurers; no guarantees of returns.
Cover, where the data is provided: loan repayment capacity (FOIR), life cover gap (HLV
protection deficit) and goal success probability from the Monte Carlo simulation.
Number rules (strict):
- Use only the figures in the input, copied exactly as written (same digits, commas, ₹ and %).
- Never calculate, round, convert, total or estimate anything. Do not introduce new numbers.
- Refer to a figure by quoting it, or describe it in words without a number.
Style: natural prose. Do not put figures in quotation marks and do not repeat the input labels
verbatim (write "the eligible home loan is ₹76,69,120", not "eligible home loan of \"76,69,120\"").
Format: 3 to 5 short paragraphs of plain prose, no headings, no tables, no disclaimer."""

MAX_ATTEMPTS = 2
SMALL_COUNTING_NUMBERS = set(range(11))
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def format_number(value: float | int) -> str:
    """Indian digit grouping (12,34,567.89); at most 2 decimals, trailing zeros dropped."""
    negative = value < 0
    text = f"{abs(value):.2f}".rstrip("0").rstrip(".")
    whole, _, frac = text.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = [head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1]
        whole = ",".join(groups + [tail])
    return ("-" if negative else "") + whole + (f".{frac}" if frac else "")


def _format_values(data: Any) -> Any:
    """Format numbers for narration; amounts of 1,000 or more are shown in whole units."""
    if isinstance(data, dict):
        return {k: _format_values(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_format_values(v) for v in data]
    if isinstance(data, bool):
        return "Yes" if data else "No"
    if not isinstance(data, (int, float)):
        return data
    return format_number(round(data) if abs(data) >= 1_000 else data)


def _render(data: dict[str, Any], indent: int = 0) -> list[str]:
    """Readable 'Label: value' lines (nested sections indented) for the prompt."""
    pad = "  " * indent
    lines = []
    for key, value in data.items():
        label = str(key).replace("_", " ").capitalize()
        if isinstance(value, dict):
            lines.append(f"{pad}{label}:")
            lines += _render(value, indent + 1)
        elif isinstance(value, list):
            lines.append(f"{pad}{label}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{pad}  -")
                    lines += _render(item, indent + 2)
                else:
                    lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{label}: {value}")
    return lines


def _numbers_in(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUMBER.findall(text)}


def untraceable_numbers(summary: str, allowed_source: str) -> set[float]:
    """Numbers in the summary that do not appear in the input figures."""
    allowed = _numbers_in(allowed_source) | _numbers_in(DISCLAIMER) | SMALL_COUNTING_NUMBERS
    return {n for n in _numbers_in(summary) if n not in allowed}


def _fallback_summary(formatted_results: dict[str, Any]) -> str:
    return "\n".join(["Summary of calculated results:", *_render(formatted_results)])


def generate_plan_explanation(
    profile: UserProfileSchema,
    results: dict[str, Any],
    client: genai.Client | None = None,
) -> str:
    """Narrative summary of engine results, always ending with the statutory disclaimer."""
    client = client or get_client()
    formatted_results = _format_values(results)
    source = "\n".join([
        "Household profile:", *_render(_format_values(profile.model_dump()), 1),
        "Calculated results:", *_render(formatted_results, 1),
    ])
    prompt = f"Write the summary using only these figures:\n{source}"

    summary = None
    for _ in range(MAX_ATTEMPTS):
        response = generate_content(
            client,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION, temperature=0.2),
        )
        text = (response.text or "").strip()
        stray = untraceable_numbers(text, source)
        if text and not stray:
            summary = text
            break
        if stray:
            prompt = (
                f"{prompt}\n\nYour previous draft used figures that are not in the input "
                f"({', '.join(format_number(n) for n in sorted(stray))}). Rewrite it using only the "
                "input figures, copied exactly."
            )

    return f"{summary or _fallback_summary(formatted_results)}\n\n{DISCLAIMER}"
