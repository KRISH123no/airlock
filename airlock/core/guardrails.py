"""Guardrails: what happens to text on the way in and on the way out.

Three separate jobs, deliberately not conflated:

**Redaction** removes personal data before it leaves your building. It is
reversible — the gateway swaps values for placeholders, the model sees the
placeholders, and the real values are put back in the response. The model
never receives the card number, and the caller never notices.

**Schema repair** deals with the most common production failure of structured
output: a model that returns *almost* JSON. Wrapped in a code fence, with a
sentence of preamble, a trailing comma. Repairing that locally is far cheaper
than a second model call.

**Injection screening** is heuristic and is honest about it. There is no
reliable detector for prompt injection; what exists are patterns that catch
the obvious attempts. It reports a score and the rules that fired so a caller
can decide, rather than pretending to a verdict.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------- redaction


@dataclass(frozen=True)
class PIIRule:
    name: str
    pattern: re.Pattern[str]
    #: Optional extra check, e.g. a checksum, to cut false positives.
    validator: Any = None


def _luhn(digits: str) -> bool:
    """The checksum every real card number satisfies.

    Without it, any 16-digit number — an order id, a timestamp pair — reads as
    a credit card, and redacting those corrupts legitimate prompts.
    """
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _is_card(match: str) -> bool:
    digits = re.sub(r"[^0-9]", "", match)
    return 13 <= len(digits) <= 19 and _luhn(digits)


DEFAULT_PII_RULES: tuple[PIIRule, ...] = (
    PIIRule("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    PIIRule("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), _is_card),
    PIIRule(
        "PHONE",
        re.compile(r"(?<![\w.])\+?\d{1,3}[\s-]?\(?\d{3,5}\)?[\s-]?\d{3}[\s-]?\d{4}\b"),
    ),
    PIIRule("SECRET", re.compile(r"\b(?:sk|pk|ghp|gho|xox[baprs])[-_][A-Za-z0-9_\-]{16,}\b")),
    PIIRule("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
)


@dataclass
class Redaction:
    text: str
    #: placeholder -> the original value it stands for.
    replacements: dict[str, str] = field(default_factory=dict)

    @property
    def found(self) -> list[str]:
        """The rule names that matched, in order of first appearance."""
        seen: list[str] = []
        for placeholder in self.replacements:
            name = placeholder.strip("[]").rsplit("_", 1)[0]
            if name not in seen:
                seen.append(name)
        return seen

    def restore(self, text: str) -> str:
        """Put the original values back into a model's response."""
        for placeholder, original in self.replacements.items():
            text = text.replace(placeholder, original)
        return text


class Redactor:
    def __init__(self, rules: tuple[PIIRule, ...] = DEFAULT_PII_RULES) -> None:
        self.rules = rules

    def redact_many(self, texts: list[str]) -> tuple[list[str], Redaction]:
        """Redact several texts against one shared placeholder map.

        Doing them independently would be wrong: two different emails in two
        messages would each be numbered `[EMAIL_1]`, and restoring would put
        the first value everywhere. One counter across the whole request keeps
        placeholders unique and the round trip exact.
        """
        replacements: dict[str, str] = {}
        counters: dict[str, int] = {}
        out = [self._redact_into(text, replacements, counters) for text in texts]
        return out, Redaction("\n".join(out), replacements)

    def redact(self, text: str) -> Redaction:
        replacements: dict[str, str] = {}
        counters: dict[str, int] = {}
        return Redaction(self._redact_into(text, replacements, counters), replacements)

    def _redact_into(
        self, text: str, replacements: dict[str, str], counters: dict[str, int]
    ) -> str:
        """Redact one text, adding to a caller-owned placeholder map."""
        for rule in self.rules:

            def substitute(match: re.Match[str], rule: PIIRule = rule) -> str:
                value = match.group(0)
                if rule.validator is not None and not rule.validator(value):
                    return value
                # The same value seen twice reuses its placeholder.
                for existing, original in replacements.items():
                    if original == value:
                        return existing
                counters[rule.name] = counters.get(rule.name, 0) + 1
                placeholder = f"[{rule.name}_{counters[rule.name]}]"
                replacements[placeholder] = value
                return placeholder

            text = rule.pattern.sub(substitute, text)
        return text


# ----------------------------------------------------------------- schema repair


class SchemaError(ValueError):
    """The output did not satisfy the requested schema."""


def extract_json(text: str) -> str:
    """Pull the JSON out of a response that wrapped it in something else."""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    start = min(
        (i for i in (stripped.find("{"), stripped.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        return stripped
    opening = stripped[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped[start:]


def repair_json(text: str) -> Any:
    """Parse JSON from a model response, fixing what is safely fixable.

    Only structural noise is repaired — fences, preamble, trailing commas,
    Python literals. Nothing here invents or reshapes data: if the content is
    wrong, that is a model problem, not a parsing problem.
    """
    candidate = extract_json(text)
    attempts = [
        candidate,
        re.sub(r",\s*([}\]])", r"\1", candidate),  # trailing commas
    ]
    literals = {"True": "true", "False": "false", "None": "null"}
    attempts.append(
        re.sub(r"\b(True|False|None)\b", lambda m: literals[m.group(0)], attempts[1])
    )
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    raise SchemaError("response was not JSON, even after repair")


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """A small JSON Schema subset: type, required, properties, items, enum.

    Enough for the structured-output case the gateway cares about, and no
    dependency. Unsupported keywords are ignored rather than guessed at.
    """
    errors: list[str] = []
    expected = schema.get("type")
    checks = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    if expected in checks:
        wanted = checks[expected]
        # bool is a subclass of int in Python; a schema asking for a number
        # should not silently accept True.
        if expected in {"number", "integer"} and isinstance(value, bool):
            errors.append(f"{path}: expected {expected}, found boolean")
            return errors
        if not isinstance(value, wanted):
            errors.append(f"{path}: expected {expected}, found {type(value).__name__}")
            return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")

    if expected == "object" or isinstance(value, dict):
        for key in schema.get("required", []):
            if not isinstance(value, dict) or key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        for key, subschema in schema.get("properties", {}).items():
            if isinstance(value, dict) and key in value:
                errors += validate(value[key], subschema, f"{path}.{key}")

    if (expected == "array" or isinstance(value, list)) and "items" in schema:
        for index, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{index}]")

    return errors


def parse_structured(text: str, schema: dict[str, Any]) -> Any:
    """Repair, parse and validate a structured response, or raise SchemaError."""
    value = repair_json(text)
    errors = validate(value, schema)
    if errors:
        raise SchemaError("; ".join(errors))
    return value


# ------------------------------------------------------------ injection screening


@dataclass(frozen=True)
class InjectionRule:
    name: str
    pattern: re.Pattern[str]
    weight: float
    why: str


INJECTION_RULES: tuple[InjectionRule, ...] = (
    InjectionRule(
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget)\b[^.\n]{0,30}"
            r"\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}"
            r"\b(instruction|prompt|rule|direction)",
            re.I,
        ),
        0.5,
        "asks the model to discard its instructions",
    ),
    InjectionRule(
        "prompt_exfiltration",
        re.compile(
            r"\b(print|reveal|show|repeat|output|display)\b[^.\n]{0,30}"
            r"\b(system prompt|initial instruction|your instruction|everything above)",
            re.I,
        ),
        0.45,
        "asks the model to disclose its system prompt",
    ),
    InjectionRule(
        "role_switch",
        re.compile(
            r"\b(you are now|from now on,? you|act as if you|pretend (to be|you are))\b", re.I
        ),
        0.3,
        "attempts to reassign the model's role",
    ),
    InjectionRule(
        "fake_turn",
        re.compile(r"(^|\n)\s*(system|assistant)\s*:", re.I),
        0.35,
        "forges a conversation turn to look like the system speaking",
    ),
    InjectionRule(
        "delimiter_injection",
        re.compile(
            r"(<\|[a-z_]+\|>|\[/?INST\]|<<SYS>>|###\s*(system|instruction))", re.I
        ),
        0.4,
        "uses chat template delimiters as text",
    ),
    InjectionRule(
        "guardrail_bypass",
        re.compile(
            r"\b(developer mode|jailbreak|DAN mode|without any (restriction|filter)|"
            r"bypass (your )?(safety|guardrail|filter))\b",
            re.I,
        ),
        0.4,
        "explicitly asks to bypass safety behaviour",
    ),
    InjectionRule(
        "encoded_payload",
        re.compile(r"\b[A-Za-z0-9+/]{120,}={0,2}\b"),
        0.25,
        "contains a long encoded blob that hides its content from inspection",
    ),
)


@dataclass
class InjectionVerdict:
    score: float
    matched: list[InjectionRule] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return self.score > 0.0

    def explain(self) -> str:
        if not self.matched:
            return "no injection patterns matched"
        return "; ".join(f"{rule.name} ({rule.why})" for rule in self.matched)


class InjectionScreen:
    """Heuristic prompt-injection detection.

    This catches the obvious attempts and will miss a careful one. It is a
    filter, not a guarantee, and anything relying on it should be designed so
    that a bypass is survivable — least privilege on tools, no secrets in the
    system prompt, output treated as untrusted.
    """

    def __init__(
        self, rules: tuple[InjectionRule, ...] = INJECTION_RULES, threshold: float = 0.5
    ) -> None:
        self.rules = rules
        self.threshold = threshold

    def inspect(self, text: str) -> InjectionVerdict:
        matched = [rule for rule in self.rules if rule.pattern.search(text)]
        # Scores are combined so that several weak signals can add up to a
        # strong one, without any single rule being able to exceed 1.
        score = 1.0
        for rule in matched:
            score *= 1.0 - rule.weight
        return InjectionVerdict(round(1.0 - score, 4), matched)

    def blocks(self, text: str) -> bool:
        return self.inspect(text).score >= self.threshold
