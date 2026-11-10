"""MLflow filter_string and order_by parser and evaluator."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from .errors import TrackerError
from .models import Run
from .validation import MAX_FILTER_LENGTH

EntityKind = Literal["metrics", "params", "tags", "attributes"]

_NUMERIC_OPS = frozenset({"=", "!=", ">", ">=", "<", "<="})
_STRING_OPS = frozenset({"=", "!=", "LIKE", "ILIKE"})
_VALID_ENTITIES = frozenset({"metrics", "params", "tags", "attributes"})
_VALID_ATTRIBUTES = frozenset({"run_name", "status", "start_time"})
_NUMERIC_ATTRIBUTES = frozenset({"start_time"})
_IDENT_KEY_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


@dataclass(frozen=True)
class FilterClause:
    """Parsed comparison clause from a filter_string."""

    entity: EntityKind
    key: str
    op: str
    value: float | str
    is_numeric: bool


@dataclass(frozen=True)
class OrderByClause:
    """Parsed ordering directive."""

    entity: EntityKind
    key: str
    ascending: bool


class _Lexer:
    """Character-by-character lexer for MLflow filter strings."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.pos = 0

    def skip_ws(self) -> None:
        while self.pos < self.n and self.text[self.pos].isspace():
            self.pos += 1

    def at_end(self) -> bool:
        self.skip_ws()
        return self.pos >= self.n

    def read_entity_and_key(self) -> tuple[EntityKind, str]:
        self.skip_ws()
        start = self.pos
        while self.pos < self.n and (self.text[self.pos].isalpha() or self.text[self.pos] == "_"):
            self.pos += 1
        entity_raw = self.text[start : self.pos]
        if entity_raw not in _VALID_ENTITIES:
            raise TrackerError("bad_filter", f"Invalid entity {entity_raw!r} in filter.")
        if self.pos >= self.n or self.text[self.pos] != ".":
            raise TrackerError("bad_filter", "Expected '.' after entity name.")
        self.pos += 1  # consume '.'
        if self.pos >= self.n:
            raise TrackerError("bad_filter", "Missing key after '.' in filter.")

        if self.text[self.pos] == "`":
            self.pos += 1
            k_start = self.pos
            while self.pos < self.n and self.text[self.pos] != "`":
                self.pos += 1
            if self.pos >= self.n:
                raise TrackerError("bad_filter", "Unterminated backtick key in filter.")
            key = self.text[k_start : self.pos]
            self.pos += 1  # consume closing '`'
            if not key:
                raise TrackerError("bad_filter", "Empty backtick-quoted key in filter.")
        else:
            k_start = self.pos
            while self.pos < self.n and (
                self.text[self.pos].isalnum() or self.text[self.pos] in "_-"
            ):
                self.pos += 1
            key = self.text[k_start : self.pos]
            if not key:
                raise TrackerError("bad_filter", "Expected key identifier after '.'.")

        entity: EntityKind = entity_raw  # type: ignore[assignment]
        if entity == "attributes" and key not in _VALID_ATTRIBUTES:
            raise TrackerError(
                "bad_filter",
                f"Unsupported attribute {key!r}: must be run_name, status, or start_time.",
            )
        return entity, key

    def read_operator(self) -> str:
        self.skip_ws()
        if self.pos >= self.n:
            raise TrackerError("bad_filter", "Missing operator in filter.")
        two = self.text[self.pos : self.pos + 2]
        if two in (">=", "<=", "!="):
            self.pos += 2
            # Check that the next char isn't another operator symbol like >> or ==
            if self.pos < self.n and self.text[self.pos] in "=<>!":
                raise TrackerError("bad_filter", f"Invalid operator near {two!r}.")
            return two
        one = self.text[self.pos]
        if one in ("=", ">", "<"):
            self.pos += 1
            if self.pos < self.n and self.text[self.pos] in "=<>!":
                raise TrackerError("bad_filter", f"Invalid operator starting with {one!r}.")
            return one
        # Word operators: LIKE or ILIKE
        start = self.pos
        while self.pos < self.n and self.text[self.pos].isalpha():
            self.pos += 1
        word = self.text[start : self.pos].upper()
        if word in ("LIKE", "ILIKE"):
            return word
        raise TrackerError("bad_filter", f"Invalid operator at position {start}.")

    def read_number(self) -> float:
        self.skip_ws()
        if self.pos >= self.n:
            raise TrackerError("bad_filter", "Missing numeric value in filter.")
        m = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", self.text[self.pos :])
        if not m:
            raise TrackerError("bad_filter", "Expected numeric literal in filter.")
        token = m.group(0)
        self.pos += len(token)
        # Next character must be whitespace or end of string
        if self.pos < self.n and not self.text[self.pos].isspace():
            raise TrackerError(
                "bad_filter", f"Unexpected trailing characters after number {token!r}."
            )
        val = float(token)
        if math.isnan(val) or math.isinf(val):
            raise TrackerError("bad_filter", "Numeric literal in filter must be finite.")
        return val

    def read_quoted_string(self) -> str:
        self.skip_ws()
        if self.pos >= self.n or self.text[self.pos] not in ("'", '"'):
            raise TrackerError("bad_filter", "Expected single- or double-quoted string value.")
        quote = self.text[self.pos]
        self.pos += 1
        chars: list[str] = []
        while self.pos < self.n:
            ch = self.text[self.pos]
            if ch == "\\":
                self.pos += 1
                if self.pos >= self.n:
                    raise TrackerError("bad_filter", "Unterminated escape in string literal.")
                chars.append(self.text[self.pos])
                self.pos += 1
            elif ch == quote:
                self.pos += 1
                if self.pos < self.n and not self.text[self.pos].isspace():
                    raise TrackerError(
                        "bad_filter", "Unexpected characters immediately after quoted string."
                    )
                return "".join(chars)
            else:
                chars.append(ch)
                self.pos += 1
        raise TrackerError("bad_filter", "Unterminated string literal in filter.")

    def read_and(self) -> None:
        self.skip_ws()
        start = self.pos
        while self.pos < self.n and self.text[self.pos].isalpha():
            self.pos += 1
        word = self.text[start : self.pos]
        if word.upper() != "AND" or (self.pos < self.n and not self.text[self.pos].isspace()):
            raise TrackerError("bad_filter", f"Expected 'AND' between clauses, got {word!r}.")


def parse_filter(filter_string: str) -> list[FilterClause]:
    """Parse an MLflow subset filter string into a list of FilterClauses."""
    if not isinstance(filter_string, str):
        raise TrackerError("bad_filter", "filter_string must be a string.")
    if len(filter_string) > MAX_FILTER_LENGTH:
        raise TrackerError(
            "bad_filter",
            f"filter_string exceeds maximum length of {MAX_FILTER_LENGTH} characters.",
        )
    if not filter_string.strip():
        return []

    lexer = _Lexer(filter_string)
    clauses: list[FilterClause] = []
    while True:
        entity, key = lexer.read_entity_and_key()
        op = lexer.read_operator()
        is_numeric = entity == "metrics" or (entity == "attributes" and key in _NUMERIC_ATTRIBUTES)
        if is_numeric:
            if op not in _NUMERIC_OPS:
                raise TrackerError(
                    "bad_filter",
                    f"Operator {op!r} is not valid for numeric field {entity}.{key}.",
                )
            num_val = lexer.read_number()
            clauses.append(
                FilterClause(entity=entity, key=key, op=op, value=num_val, is_numeric=True)
            )
        else:
            if op not in _STRING_OPS:
                raise TrackerError(
                    "bad_filter",
                    f"Operator {op!r} is not valid for string field {entity}.{key}.",
                )
            str_val = lexer.read_quoted_string()
            clauses.append(
                FilterClause(entity=entity, key=key, op=op, value=str_val, is_numeric=False)
            )

        if lexer.at_end():
            break
        lexer.read_and()
        if lexer.at_end():
            raise TrackerError("bad_filter", "Trailing 'AND' in filter_string.")

    return clauses


def _sql_like_to_regex(pattern: str, case_insensitive: bool) -> re.Pattern[str]:
    """Convert a SQL LIKE pattern with '%' and '_' wildcards into a compiled regex."""
    parts: list[str] = ["^"]
    for ch in pattern:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    parts.append("$")
    flags = re.DOTALL | (re.IGNORECASE if case_insensitive else 0)
    return re.compile("".join(parts), flags)


def _extract_field(run: Run, entity: EntityKind, key: str) -> float | str | None:
    """Extract a field value from a Run, or None if missing."""
    if entity == "metrics":
        return run.data.metrics.get(key)
    if entity == "params":
        return run.data.params.get(key)
    if entity == "tags":
        return run.data.tags.get(key)
    if key == "run_name":
        return run.info.run_name
    if key == "status":
        return run.info.status
    if key == "start_time":
        return float(run.info.start_time)
    return None


def matches_filter(run: Run, clauses: list[FilterClause]) -> bool:
    """Return True if a Run satisfies all FilterClauses."""
    for clause in clauses:
        actual = _extract_field(run, clause.entity, clause.key)
        if actual is None:
            return False
        if clause.is_numeric:
            f_actual = float(actual)
            if math.isnan(f_actual):
                return False
            f_target = float(clause.value)
            if clause.op == "=" and f_actual != f_target:
                return False
            if clause.op == "!=" and f_actual == f_target:
                return False
            if clause.op == ">" and not (f_actual > f_target):
                return False
            if clause.op == ">=" and not (f_actual >= f_target):
                return False
            if clause.op == "<" and not (f_actual < f_target):
                return False
            if clause.op == "<=" and not (f_actual <= f_target):
                return False
        else:
            s_actual = str(actual)
            s_target = str(clause.value)
            if clause.op == "=" and s_actual != s_target:
                return False
            if clause.op == "!=" and s_actual == s_target:
                return False
            if clause.op == "LIKE" and not _sql_like_to_regex(s_target, False).match(s_actual):
                return False
            if clause.op == "ILIKE" and not _sql_like_to_regex(s_target, True).match(s_actual):
                return False
    return True


def parse_order_by(order_by: list[str] | None) -> list[OrderByClause]:
    """Parse a list of order_by strings into OrderByClauses."""
    if order_by is None:
        return []
    if not isinstance(order_by, list):
        raise TrackerError("bad_filter", "order_by must be a list of strings.")
    result: list[OrderByClause] = []
    for item in order_by:
        if not isinstance(item, str) or not item.strip() or len(item) > MAX_FILTER_LENGTH:
            raise TrackerError("bad_filter", "Invalid order_by item.")
        lexer = _Lexer(item)
        entity, key = lexer.read_entity_and_key()
        lexer.skip_ws()
        ascending = True
        if not lexer.at_end():
            start = lexer.pos
            while lexer.pos < lexer.n and lexer.text[lexer.pos].isalpha():
                lexer.pos += 1
            direction = lexer.text[start : lexer.pos].upper()
            if direction == "ASC":
                ascending = True
            elif direction == "DESC":
                ascending = False
            else:
                raise TrackerError(
                    "bad_filter", f"Invalid sort direction {direction!r} in order_by."
                )
            if not lexer.at_end():
                raise TrackerError(
                    "bad_filter", f"Unexpected trailing tokens in order_by {item!r}."
                )
        result.append(OrderByClause(entity=entity, key=key, ascending=ascending))
    return result


class _ReverseComparable:
    """Helper wrapper to reverse comparison order for DESC sorting."""

    __slots__ = ("val",)

    def __init__(self, val: float | str) -> None:
        self.val = val

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _ReverseComparable):
            return NotImplemented
        return bool(self.val > other.val)  # type: ignore[operator]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ReverseComparable):
            return NotImplemented
        return bool(self.val == other.val)


def sort_runs(runs: list[Run], order_clauses: list[OrderByClause]) -> list[Run]:
    """Sort runs by order_clauses with nulls/NaNs last.

    Tie-breaking by start_time DESC, run_id ASC.
    """

    def sort_key(run: Run) -> tuple[object, ...]:
        parts: list[object] = []
        for clause in order_clauses:
            val = _extract_field(run, clause.entity, clause.key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                parts.append((1, 0))
            else:
                comp: object = val if clause.ascending else _ReverseComparable(val)
                parts.append((0, comp))
        # Default tie-breaker: start_time DESC, run_id ASC
        parts.append((-run.info.start_time, run.info.run_id))
        return tuple(parts)

    return sorted(runs, key=sort_key)
