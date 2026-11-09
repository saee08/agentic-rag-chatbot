"""Deterministic arithmetic for generated answers.

Language models make arithmetic mistakes (observed: "11 × 6" answered as 72). The answer prompt
therefore tells the model never to state a computed number itself but to write
``[[calc: <expression>]]``; this module evaluates those expressions exactly. As a second safety
net, explicit equations such as "11 × 6 = 72" are recomputed and corrected.

Only numbers, + - * / and parentheses are accepted (parsed with ``ast``; nothing is executed).
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from decimal import Decimal

CALC_RE = re.compile(r"\[\[\s*calc\s*:\s*([^\]]{1,200})\]\]", re.IGNORECASE)
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_EQUATION_RE = re.compile(
    rf"(?P<lhs>(?:{_NUMBER})(?:\s*[×x*/÷+\-]\s*(?:{_NUMBER}))+)\s*=\s*(?P<rhs>{_NUMBER})(?![\d.,]*\d)"
)
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_MAX_ABS = Decimal(10) ** 15


class CalcError(ValueError):
    pass


def _normalise(expression: str) -> str:
    expr = expression.replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", expr)  # "11 x 6"
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)  # "1,850" -> "1850"


def evaluate(expression: str) -> Decimal:
    try:
        tree = ast.parse(_normalise(expression).strip(), mode="eval")
    except SyntaxError as exc:
        raise CalcError("not an arithmetic expression") from exc

    def walk(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            value = Decimal(str(node.value))
            if abs(value) > _MAX_ABS:
                raise CalcError("number out of range")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise CalcError("division by zero")
            result = _OPS[type(node.op)](left, right)
            if abs(result) > _MAX_ABS:
                raise CalcError("result out of range")
            return result
        raise CalcError("unsupported element in expression")

    return walk(tree)


def format_number(value: Decimal) -> str:
    rounded = value.quantize(Decimal("0.01")).normalize()
    if rounded == rounded.to_integral():
        return f"{int(rounded):,}"
    return f"{rounded:,f}"


@dataclass(frozen=True)
class CalcResult:
    text: str
    calculations: int  # [[calc: ...]] expressions evaluated
    corrections: int  # explicit equations whose stated result was wrong and has been fixed
    failures: int  # expressions that could not be evaluated (left as written, without the markup)


def apply_calculations(text: str) -> CalcResult:
    calculations = failures = corrections = 0

    def _calc(match: re.Match[str]) -> str:
        nonlocal calculations, failures
        try:
            value = evaluate(match.group(1))
        except CalcError:
            failures += 1
            return match.group(1).strip()
        calculations += 1
        return format_number(value)

    text = CALC_RE.sub(_calc, text)

    def _equation(match: re.Match[str]) -> str:
        nonlocal corrections
        try:
            expected = evaluate(match.group("lhs"))
            stated = Decimal(match.group("rhs").replace(",", ""))
        except (CalcError, ArithmeticError):
            return match.group(0)
        if abs(expected - stated) <= Decimal("0.01"):
            return match.group(0)
        corrections += 1
        return match.group(0)[: match.start("rhs") - match.start()] + format_number(expected)

    text = _EQUATION_RE.sub(_equation, text)
    return CalcResult(text, calculations, corrections, failures)
