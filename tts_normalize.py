# -*- coding: utf-8 -*-
"""
Korean TTS-friendly math text normalizer (rule-based, recursive).

핵심 규칙:
- 프라임: ' → 프라임, '' → 더블 프라임, ''' → 트리플 프라임
- + → 플러스, = → 는, (이항) - → 마이너스
- 지수: a^{b}, a^b → "a의 b승"
  * 단, base가 e/E이면 "e의 b" (승 생략). b가 음수면 "마이너스 …"
  * 지수 내용은 재귀적으로 정규화
- 분수: \frac{a}{b}, a/b → "a 나누기 b"
- 괄호/브래킷 ()[]{} 기호는 삭제(내용은 유지) — 삭제는 항상 마지막에 수행
- 서브스크립트 x_y, x_{y} → 언더스코어 제거(읽지 않음)
- 함수: sin/cos/tan/sinh/cosh/tanh/ln/log/exp → 사인/코사인/…/자연로그/로그/지수함수
- 그리스 문자(일부): alpha,beta,gamma,lambda,mu,pi → 알파/베타/…
- 화살표·함축: \Rightarrow, ⇒, →, =>, \implies → 따라서
- 절댓값: |x| → 절댓값 x
- 부등호:
  ≤, \le(=) → 이하 /  ≥, \ge(=) → 이상
  <, \lt → 보다 작다 / >, \gt → 보다 크다
- 근사: ≈, \approx, \simeq → 약 (또는 대략)
- 시그마/적분: ∑, \sum → 시그마 합 / ∫, \int → 적분
- 미분연산자: d/dt, d/dx → "디 티에 대한 미분", "디 엑스에 대한 미분"
- 기타 LaTeX 토큰: \left, \right, \, \! \; \: \quad \qquad \displaystyle \cdot → 제거/공백화
- \text{...} → 내용만 남김

사용:
    python tts_normalize.py --in input.txt --out output.txt
    echo "y' = y + e^{-kt}" | python tts_normalize.py
"""
from __future__ import annotations
import argparse
import re
import sys

# ---------- helpers ----------

SPACE_COLLAPSE = re.compile(r"\s{2,}")

def collapse_spaces(s: str) -> str:
    return SPACE_COLLAPSE.sub(" ", s).strip()

def replace_leading_minus_to_korean(s: str) -> str:
    # 지수 내부 맨 앞의 '-'를 '마이너스 '로 바꿔줌
    return re.sub(r"^\s*-\s*", "마이너스 ", s)

# ---------- core normalizer (recursive on fragments) ----------

def normalize_math_fragment(s: str) -> str:
    """수식 조각(지수, 분자/분모, 괄호 내부 등)에 재귀 규칙 적용"""
    return _normalize_line(s, recursive=True)

# ---------- regexes ----------

PRIME_PATTERNS = [
    (re.compile(r"([A-Za-z가-힣0-9])'''"), r"\1 트리플 프라임"),
    (re.compile(r"([A-Za-z가-힣0-9])''"), r"\1 더블 프라임"),
    (re.compile(r"([A-Za-z가-힣0-9])'"), r"\1 프라임"),
]

# Exponent: a^{b}, a^b  (나중에 base=e/E는 특수 처리)
EXP_BRACED = re.compile(r"([A-Za-z가-힣0-9πeE])\^\{([^{}]+)\}")
EXP_SIMPLE = re.compile(r"([A-Za-z가-힣0-9πeE])\^([A-Za-z가-힣0-9+\-/*|\\]+)")

# Subscripts: drop underscore
SUB_BRACED = re.compile(r"_[{]([^{}]+)[}]")
SUB_SIMPLE = re.compile(r"_([A-Za-z0-9]+)")

# Fractions
FRAC_LATEX = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
FRAC_SLASH = re.compile(r"\b([A-Za-z0-9]+)\s*/\s*([A-Za-z0-9]+)\b")

# Arrows / implication
ARROWS = re.compile(r"\\Rightarrow|⇒|→|=>|\\implies")

# Functions (order matters: longer names first)
FUNC_MAP = [
    (re.compile(r"\\tanh|\btanh\b"), "쌍곡탄젠트"),
    (re.compile(r"\\sinh|\bsinh\b"), "쌍곡사인"),
    (re.compile(r"\\cosh|\bcosh\b"), "쌍곡코사인"),
    (re.compile(r"\\sin|\bsin\b"), "사인"),
    (re.compile(r"\\cos|\bcos\b"), "코사인"),
    (re.compile(r"\\tan|\btan\b"), "탄젠트"),
    (re.compile(r"\\ln|\bln\b"), "자연로그"),
    (re.compile(r"\\log|\blog\b"), "로그"),
    (re.compile(r"\\exp|\bexp\b"), "지수함수"),
]

# Greek letters
GREEK_MAP = [
    (re.compile(r"\\alpha|\balpha\b"), "알파"),
    (re.compile(r"\\beta|\bbeta\b"), "베타"),
    (re.compile(r"\\gamma|\bgamma\b"), "감마"),
    (re.compile(r"\\lambda|\blambda\b"), "람다"),
    (re.compile(r"\\mu|\bmu\b"), "뮤"),
    (re.compile(r"\\pi|\bpi\b"), "파이"),
]

# Inequalities & approx
INEQ_MAP = [
    (re.compile(r"\\leqslant|\\leq|≤"), "이하"),
    (re.compile(r"\\geqslant|\\geq|≥"), "이상"),
    (re.compile(r"\\lt|<"), "보다 작다"),
    (re.compile(r"\\gt|>"), "보다 크다"),
]
APPROX_MAP = [
    (re.compile(r"\\approx|\\simeq|≈|≃"), "약"),
]

# Sigma / Integral
SIGMA_PAT = re.compile(r"\\sum|∑")
INT_PAT = re.compile(r"\\int|∫")

# d/dt, d/dx ...
DIFF_PAT = re.compile(r"\bd/d([a-zA-Z])\b")

# LaTeX cleanups & text
LATEX_CLEANUPS = [
    (re.compile(r"\\left|\\right|\\,|\\!|\\;|\\:|\\quad|\\qquad|\\displaystyle|\\cdot"), " "),
    (re.compile(r"\\text\s*\{([^{}]*)\}"), lambda m: m.group(1)),
]

# Absolute value
ABS_PAT = re.compile(r"\|([^|]+)\|")

# Operators and symbols (order matters)
EQ_PAT = re.compile(r"(?<![<≥≤>])=+(?![<≥≤>])")  # '=' not part of <=, >= etc
PLUS_PAT = re.compile(r"\+")
# binary minus between word/digit (avoid negative sign directly after ^ handled separately)
BINARY_MINUS = re.compile(r"(?<=\w)-(?!\d)")

# Remove brackets (keep content) — run at the very end
REMOVE_BRACKETS = re.compile(r"[()\[\]{}]")

# ---------- normalization pipeline ----------

def _normalize_line(line: str, recursive: bool = False) -> str:
    s = line

    # 0) LaTeX cleanups & \text{...}
    for pat, repl in LATEX_CLEANUPS:
        s = pat.sub(repl if isinstance(repl, str) else repl, s)

    # 1) Fractions (normalize subfragments recursively)
    def frac_latex_repl(m):
        num = normalize_math_fragment(m.group(1))
        den = normalize_math_fragment(m.group(2))
        return f"{num} 나누기 {den}"
    s = FRAC_LATEX.sub(frac_latex_repl, s)

    s = FRAC_SLASH.sub(lambda m: f"{m.group(1)} 나누기 {m.group(2)}", s)

    # 2) Exponents (with special rule for base e/E)
    def exp_braced_repl(m):
        base = m.group(1)
        exp = normalize_math_fragment(m.group(2))
        exp = replace_leading_minus_to_korean(exp)
        if base in ("e", "E"):
            return f"e의 {exp}"
        return f"{base}의 {exp}승"
    s = EXP_BRACED.sub(exp_braced_repl, s)

    def exp_simple_repl(m):
        base = m.group(1)
        exp = normalize_math_fragment(m.group(2))
        exp = replace_leading_minus_to_korean(exp)
        if base in ("e", "E"):
            return f"e의 {exp}"
        return f"{base}의 {exp}승"
    s = EXP_SIMPLE.sub(exp_simple_repl, s)

    # 3) Primes
    for pat, repl in PRIME_PATTERNS:
        s = pat.sub(repl, s)

    # 4) Subscripts: drop underscore (keep content or drop entirely?)
    #   요구: '_'는 읽지 않는다 → 언더스코어 및 괄호는 제거. 내용은 유지(이미 괄호는 마지막에 제거)
    s = SUB_BRACED.sub(lambda m: m.group(1), s)  # 내용 유지
    s = SUB_SIMPLE.sub(lambda m: m.group(1), s)

    # 5) Arrows / implication
    s = ARROWS.sub("따라서", s)

    # 6) Functions & Greek
    for pat, repl in FUNC_MAP:
        s = pat.sub(repl, s)
    for pat, repl in GREEK_MAP:
        s = pat.sub(repl, s)

    # 7) Absolute value
    s = ABS_PAT.sub(lambda m: f"절댓값 {normalize_math_fragment(m.group(1))}", s)

    # 8) Inequalities & approx
    for pat, repl in INEQ_MAP:
        s = pat.sub(repl, s)
    for pat, repl in APPROX_MAP:
        s = pat.sub(repl, s)

    # 9) Sigma / Integral
    s = SIGMA_PAT.sub("시그마 합", s)
    s = INT_PAT.sub("적분", s)

    # 10) d/dt, d/dx ...
    s = DIFF_PAT.sub(lambda m: f"디 {m.group(1).upper()}에 대한 미분", s)

    # 11) Operators (=, +, binary -)
    s = EQ_PAT.sub(" 는 ", s)
    s = PLUS_PAT.sub(" 플러스 ", s)
    s = BINARY_MINUS.sub(" 마이너스 ", s)

    # 12) 마지막에 괄호류 제거(안의 내용은 이미 처리됨)
    s = REMOVE_BRACKETS.sub("", s)

    # 13) 공백 정리
    s = collapse_spaces(s)
    return s

def normalize_text(text: str) -> str:
    return "\n".join(_normalize_line(ln) for ln in text.splitlines())

# ---------- CLI ----------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Normalize math text for Korean TTS (rule-based, recursive)")
    ap.add_argument("--in", dest="infile", help="Input file (default: stdin)")
    ap.add_argument("--out", dest="outfile", help="Output file (default: stdout)")
    args = ap.parse_args(argv)

    if args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            src = f.read()
    else:
        src = sys.stdin.read()

    out = normalize_text(src)

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(out + ("\n" if not out.endswith("\n") else ""))
    else:
        sys.stdout.write(out)

if __name__ == "__main__":
    main()