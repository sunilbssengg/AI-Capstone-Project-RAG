"""
Reliability & safety controls for the agentic RAG pipeline.

This module is intentionally dependency-free (pure Python + regex) so it
can run as a fast pre/post filter around every agent call, without adding
another network hop or model call to the critical path.

It implements four layers:
  1. Input validation   -> reject malformed / empty / oversized queries
  2. Prompt-injection detection -> flag attempts to hijack the system prompt
  3. Unsafe-content screening   -> refuse clearly unsafe requests up front
  4. Output guardrails   -> redact PII, flag ungrounded / low-confidence
     answers before they reach the user

None of these are a substitute for a production moderation API — they're
lightweight, explainable heuristics appropriate for a capstone project.
Swap in a hosted moderation endpoint (e.g. Google's Text Moderation API)
for anything user-facing at scale.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_QUERY_CHARS = 2000
MIN_QUERY_CHARS = 1

# --------------------------------------------------------------------------
# 1. Input validation
# --------------------------------------------------------------------------


class InputValidationError(ValueError):
  """Raised when user input fails basic sanity checks."""


def validate_user_input(query: str) -> str:
  """
  Validates and normalizes a raw user query.

  Raises InputValidationError with a user-facing message on failure.
  Returns the cleaned query string on success.
  """
  if query is None:
    raise InputValidationError("Question cannot be empty.")

  cleaned = query.strip()

  # Strip non-printable / control characters (defends against terminal or
  # markdown-injection tricks hidden in invisible unicode).
  cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch in "\n\t")

  if len(cleaned) < MIN_QUERY_CHARS:
    raise InputValidationError("Question cannot be empty.")

  if len(cleaned) > MAX_QUERY_CHARS:
    raise InputValidationError(
        f"Question is too long ({len(cleaned)} chars). "
        f"Please limit it to {MAX_QUERY_CHARS} characters."
    )

  return cleaned


# --------------------------------------------------------------------------
# 2. Prompt-injection detection
# --------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above)\s*(instructions|prompts|rules)",
    r"disregard (all|any|the)?\s*(previous|prior|above)\s*(instructions|prompts|rules)",
    r"you are now",
    r"act as (if you (are|were))?",
    r"reveal (your|the)\s*(system prompt|instructions|guidelines)",
    r"repeat (your|the)\s*(system prompt|instructions)",
    r"what (are|is) your (system prompt|instructions)",
    r"pretend (you|to)",
    r"jailbreak",
    r"developer mode",
    r"do anything now",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(query: str) -> bool:
  """Heuristic check for common prompt-injection / jailbreak phrasing."""
  return bool(_INJECTION_RE.search(query))


# --------------------------------------------------------------------------
# 3. Unsafe-content screening (pre-generation)
# --------------------------------------------------------------------------

_UNSAFE_PATTERNS = [
    r"\bhow (do|can) i (make|build|synthesi[sz]e)\b.*\b(bomb|explosive|nerve agent|virus|malware|ransomware)\b",
    r"\bhow (do|can) i (hack|exploit|breach)\b",
    r"\bhow (do|can) i (kill|harm|hurt)\b.*\b(myself|someone|him|her|them)\b",
    r"\bwrite (a |me )?(malware|ransomware|virus|keylogger)\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


def detect_unsafe_request(query: str) -> bool:
  """
  Heuristic pre-generation screen for clearly unsafe requests (weapons,
  self-harm, malware, hacking). This is a coarse keyword/regex net, not a
  full safety classifier — it exists to catch obvious cases before they
  ever reach the LLM or the document retriever.
  """
  return bool(_UNSAFE_RE.search(query))


REFUSAL_MESSAGE = (
    "I can't help with that request. I'm scoped to answering questions "
    "about the documents you've uploaded — feel free to ask about their "
    "contents instead."
)


# --------------------------------------------------------------------------
# 4. Output guardrails
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

_NO_ANSWER_MARKERS = (
    "i don't know",
    "i do not know",
    "don't have information",
    "do not have information",
    "no relevant information",
    "not mentioned in",
    "couldn't find",
    "could not find",
)


def redact_pii(text: str) -> str:
  """Redacts common PII patterns (emails, phone numbers, SSNs, card
  numbers) from a response before it's shown to the user. This protects
  against sensitive data embedded in source documents leaking verbatim
  into the chat UI or logs."""
  text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
  text = _SSN_RE.sub("[REDACTED_SSN]", text)
  text = _CC_RE.sub("[REDACTED_CARD_NUMBER]", text)
  text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
  return text


def is_grounded(answer: str, retrieved_docs: list) -> bool:
  """
  Heuristic grounding check: an answer is considered "grounded" if either
  (a) the model explicitly says it doesn't know / found nothing, which is
  the honest, desired behavior when retrieval comes up empty, or
  (b) at least one chunk was actually retrieved for the model to draw on.

  This won't catch subtle hallucination inside an otherwise-grounded
  answer, but it reliably catches the highest-risk case: the model
  fabricating an answer when the retriever found nothing at all.
  """
  lowered = answer.lower()
  if any(marker in lowered for marker in _NO_ANSWER_MARKERS):
    return True
  return bool(retrieved_docs)


@dataclass
class GuardedResponse:
  """Result of running output guardrails over a raw agent answer."""

  text: str
  warnings: list[str] = field(default_factory=list)


def apply_output_guardrails(answer: str, retrieved_docs: list) -> GuardedResponse:
  """Runs all output-side checks and returns the sanitized text plus any
  warnings to surface in the UI."""
  warnings: list[str] = []

  sanitized = redact_pii(answer)
  if sanitized != answer:
    warnings.append(
        "Potential PII was detected in the response and has been redacted."
    )

  if not is_grounded(sanitized, retrieved_docs):
    warnings.append(
        "This answer wasn't clearly grounded in retrieved document "
        "content — treat it with extra caution and verify against the "
        "source documents."
    )

  return GuardedResponse(text=sanitized, warnings=warnings)
