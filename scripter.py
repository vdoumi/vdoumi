"""
scripter.py
-----------
Consumes solver output (solutions-<chapter>.json) and produces a **slide-centric** lecture plan:
each slide contains (1) the MARP slide content, (2) the narration lines to speak **while that slide is shown**,
and (3) Manim blocks to overlay during that slide. This keeps visuals + speech synchronized.

Usage
-----
  python scripter.py \
    --solutions out/solutions/solutions-1.2.json \
    --outdir out/scripts \
    --model gpt-5-mini \
    --min-seconds 90 \
    --max-seconds 600 \
    --seconds-per-1kchars 150 \
    --max-retries 10 \
    --backoff 1.6 \
    --verbose

Output schema (STRICT JSON from the model)
------------------------------------------
{
  "chapter": "1.2",
  "slides": [
    {
      "id": "p5-1",
      "problem": "1.2-5",                 // omit or null for intro/outro slides
      "role": "statement|derivation|result|review|ref",  // optional label
      "marp_md": "---\n# 제목\n수식 예: $y'=ay-b$ ...",  // Full MARP slide Markdown (may include multiple equations)
      "narration": [                      // Lines spoken while THIS slide is visible (ordered)
        {"text": "여기서는 ...를 정의합니다.", "gestures": ["point"], "duration_sec": 4},
        {"text": "이제 $y(t)=Ce^{at}+\frac{b}{a}$ 를 도출합니다.", "gestures": ["write","emphasize"]}
      ],
      "manim": [                          // Optional overlays rendered via Manim during THIS slide
        {
          "id": "p5a",
          "description": "해 공간의 지수 성장 곡선과 평형선",
          "code": "from manim import *\nclass P5a(Scene):\n    def construct(self):\n        title = Text('Solution growth')\n        self.play(Write(title))\n        ax = Axes(x_range=[0,6], y_range=[-1,5])\n        curve = ax.plot(lambda x: 1.5*exp(0.5*x), color=BLUE)\n        self.play(Create(ax), Create(curve))\n        eq = MathTex(r\"y(t)=Ce^{at}+\\frac{b}{a}\")\n        eq.to_corner(UR)\n        self.play(Write(eq))\n",
          "overlay": { "position": "bottom-right", "width": "40%", "start_at_sec": 6 }
        }
      ]
    },
    ...
  ]
}

Notes
-----
- Slides are the **timeline**; speak `narration[]` in order while showing the corresponding `marp_md`.
- Include multiple equations per slide when helpful to show step-to-step flow; avoid “one equation per page.”
- Always include a final **answer slide** per problem with the key result highlighted (e.g., LaTeX `\\boxed{...}`).
- If Manim only shows graphics, also render key formulas using `MathTex` inside the Manim block so equations are present on-screen.
- Manim clips are saved as separate `.py` files and later composited **on top of** the MARP slide in post-production, per `overlay`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from typing import Dict, List, Optional

try:
  from openai import OpenAI  # type: ignore
except Exception:
  OpenAI = None  # type: ignore

SCHEMA_VERSION = "slideplan.v1"
ALLOWED_GESTURES = ["point","emphasize","small-nod","smile","pause","write","think"]

VERBOSE = False
def _log(msg: str) -> None:
  if VERBOSE:
    print(f"[SCRIPTER] {msg}", file=sys.stderr)

def _sleep_with_hint(exc: Exception, attempt: int, base: float) -> None:
  msg = str(exc)
  m = re.search(r"([0-9]*\.?[0-9]+)s", msg)
  if m:
    t = float(m.group(1))
  else:
    t = (base ** attempt) + random.uniform(0, 0.5)
  _log(f"Rate limited; sleeping {t:.2f}s...")
  time.sleep(t)

def _ensure_openai():
  if OpenAI is None:
    raise RuntimeError("openai package not installed. `pip install openai>=1.30`")
  return OpenAI()

def _read_json(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

def _write_json(path: str, data: dict) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


def _write_text(path: str, text: str) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write(text)

# Code normalization helper for Manim code blocks
def _normalize_manim_code(raw: str, target: str = "manimgl") -> str:
  """Turn JSON-style escaped code into runnable .py and port to manimgl if requested.
  - Convert literal escape sequences (\\n not preceding a letter → newline, \\t → tab, \\\\ → \\).
  - If target == 'manimgl', rewrite imports and common API calls to 3b1b/manimgl style.
  """
  code = raw.replace("\r\n", "\n")
  # Convert literal backslash-n that are *not* LaTeX commands like \neq, \nabla
  code = re.sub(r"\\n(?![A-Za-z])", "\n", code)
  code = code.replace("\\t", "\t")
  # Reduce double backslashes (JSON escaping) to single for Python source strings
  code = code.replace("\\\\", "\\")

  if target == "manimgl":
    # Import port: manimce -> manimgl
    code = code.replace("from manim import *", "from manimlib.imports import *")
    # Class/constructors: MathTex -> Tex (manimgl uses Tex/TextMobject)
    code = re.sub(r"\bMathTex\(", "Tex(", code)
    # Animations: Create(...) -> ShowCreation(...)
    code = re.sub(r"\bCreate\(", "ShowCreation(", code)
    # Some CE-only helpers may not exist; keep them if present but users can adjust.
  return code

def _assemble_marp(plan: dict) -> str:
  """Join MARP content from each slide into a single deck."""
  slides: List[str] = []
  for slide in plan.get("slides", []) or []:
    md = str(slide.get("marp_md", "")).strip()
    if md:
      slides.append(md)
  return "\n---\n".join(slides)

def _emit_manim_blocks(plan: dict, outdir: str) -> List[str]:
  """Write each manim block into its own .py file. Returns list of paths."""
  paths: List[str] = []
  base = os.path.join(outdir, "manim")
  os.makedirs(base, exist_ok=True)
  for slide in plan.get("slides", []) or []:
    sid = slide.get("id", "slide")
    for blk in (slide.get("manim") or []):
      bid = blk.get("id","a")
      fn = os.path.join(base, f"{sid}-{bid}.py")
      code_src = blk.get("code", "")
      code_src = _normalize_manim_code(code_src, target="manimgl")
      _write_text(fn, code_src)
      paths.append(fn)
  return paths

def _estimate_target_seconds(solution_markdown: str, min_s: int, max_s: int, per_1k: int) -> int:
  text = (solution_markdown or "").strip()
  n = max(1, len(text))
  est = int((n / 1000.0) * per_1k)
  return max(min_s, min(max_s, est))


def _build_prompt_for_problem(chapter: str, prob_key: str, meta: dict, target_seconds: int) -> List[dict]:
  sol = (meta.get("solution_markdown","") or "").strip()
  sys_text = (
    "You are a lecture scriptwriter for an educational VTuber. Produce a SLIDE‑CENTRIC plan in Korean (존댓말) "
    "for ONE problem only, bundling per slide: MARP Markdown, narration (with gestures), and optional Manim blocks.\n"
    "OUTPUT STRICT JSON ONLY (no code fences), escaped properly.\n\n"
    "Return JSON that matches this schema EXACTLY:\n"
    "{\n"
    f"  \"schema_version\": \"{SCHEMA_VERSION}\",\n"
    "  \"chapter\": string,\n"
    "  \"slides\": [\n"
    "    {\n"
    "      \"id\": string,                              // e.g., \"p5-1\", \"p5-2\"\n"
    "      \"problem\": string,                         // MUST be the current problem key (e.g., \"1.2-5\")\n"
    "      \"role\": \"statement\"|\"derivation\"|\"result\"|\"review\"|\"ref\"|null,\n"
    "      \"marp_md\": string,                         // FULL MARP slide markdown; MUST start with \"---\\n\"\n"
    "      \"narration\": [                             // spoken while THIS slide is visible\n"
    "        { \"text\": string, \"gestures\": [string], \"duration_sec\": number OPTIONAL }\n"
    "      ],\n"
    "      \"manim\": [                                 // optional overlay clips rendered during THIS slide\n"
    "        {\n"
    "          \"id\": string,\n"
    "          \"description\": string,\n"
    "          \"code\": string,                        // minimal runnable Manim stub; escape backslashes\n"
    "          \"overlay\": {                           // where/how to place the rendered video on top of the slide\n"
    "            \"position\": \"top-left\"|\"top-right\"|\"bottom-left\"|\"bottom-right\",\n"
    "            \"width\": string,                     // e.g., \"40%\"\n"
    "            \"start_at_sec\": number OPTIONAL\n"
    "          }\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    f"- Gesture tags must be chosen only from: {ALLOWED_GESTURES}.\n"
    "- Use multiple equations per slide when needed to show step transitions; do NOT split every equation into a new slide.\n"
    "- Every marp_md MUST begin with ---\\n. If not, the output will be invalid.\n"
    "- Never omit the leading slide delimiter.\n"
    "- Provide 2–6 slides depending on complexity; always include a final ANSWER slide with LaTeX \\\\boxed{...}.\n"
    "- Manim blocks must target **manimgl (3Blue1Brown)** style: `from manimlib.imports import *`, prefer `Tex(...)` over `MathTex(...)`, and use `ShowCreation(...)` instead of `Create(...)`.\n"
    "- If you still output CE-style (from manim import *), keep LaTeX as raw strings; newlines must be real (not literal `\\n`).\n"
    "- Keep narration concise, natural Korean, referencing what is on the slide (avoid essay tone).\n"
    "- Escape ALL backslashes in LaTeX (e.g., \\\\frac, \\\\boxed) and newlines as \\\\n inside JSON strings.\n"
    f"- Target total narration time for THIS problem: ~{target_seconds} seconds (use optional duration_sec to help meet target)."
  )
  user_text = (
    f"Chapter {chapter}, Problem {prob_key}. Here is the solved solution text:\n\n{sol}\n\n"
    "Return ONLY the JSON object with `schema_version`, `chapter`, and `slides` for this problem. "
    "Each slide's narration lines may include optional `duration_sec` to help meet the target time. "
    "Do NOT include other problems."
  )
  return [
    {"role": "system", "content": [{"type": "text", "text": sys_text}]},
    {"role": "user", "content": [{"type": "text", "text": user_text}]},
  ]


def _merge_plans(plans: List[dict], chapter: str) -> dict:
  merged = {"schema_version": SCHEMA_VERSION, "chapter": chapter, "slides": []}
  seen_ids = set()
  for plan in plans:
    for s in plan.get("slides", []) or []:
      sid = s.get("id") or "slide"
      # Ensure unique slide ids when stitching
      base = sid
      k = 1
      while sid in seen_ids:
        sid = f"{base}-{k}"
        k += 1
      s["id"] = sid
      seen_ids.add(sid)
      merged["slides"].append(s)
  return merged

def _parse_json_lenient(content: str) -> dict:
  """
  Best-effort JSON parsing:
  1) Try json.loads(content).
  2) If that fails, extract the first {...} block and try again.
  3) If we hit 'Invalid \\escape' (common with LaTeX like \\frac), escape all backslashes
     that are not part of valid JSON escapes.
  """
  try:
    return json.loads(content)
  except Exception:
    pass
  m = re.search(r"\{[\s\S]*\}", content)
  if not m:
    raise RuntimeError("Model did not return JSON.")
  s = m.group(0)
  try:
    return json.loads(s)
  except json.JSONDecodeError as e:
    if "Invalid \\escape" in str(e):
      # Escape any backslash not followed by a valid JSON escape char
      s_fixed = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", s)
      return json.loads(s_fixed)
    raise

def _validate_plan(plan: dict) -> None:
  if not isinstance(plan, dict):
    raise RuntimeError("Plan is not a JSON object.")
  if plan.get("schema_version") != SCHEMA_VERSION:
    raise RuntimeError(f"schema_version mismatch or missing (expected {SCHEMA_VERSION}).")
  if "chapter" not in plan or not isinstance(plan["chapter"], str) or not plan["chapter"].strip():
    raise RuntimeError("chapter must be a non-empty string.")
  slides = plan.get("slides")
  if not isinstance(slides, list) or not slides:
    raise RuntimeError("slides must be a non-empty array.")
  for i, s in enumerate(slides):
    if not isinstance(s, dict):
      raise RuntimeError(f"slide[{i}] must be object.")
    for k in ["id","marp_md","narration","manim"]:
      if k not in s:
        raise RuntimeError(f"slide[{i}] missing key: {k}")
    if not isinstance(s["id"], str) or not s["id"].strip():
      raise RuntimeError(f"slide[{i}].id must be non-empty string.")
    if s.get("problem") is not None and not isinstance(s.get("problem"), str):
      raise RuntimeError(f"slide[{i}].problem must be string or null.")
    role = s.get("role")
    if role is not None and role not in ["statement","derivation","result","review","ref"]:
      raise RuntimeError(f"slide[{i}].role invalid: {role}")
    if not isinstance(s["marp_md"], str) or not s["marp_md"].startswith("---\n"):
      raise RuntimeError(f"slide[{i}].marp_md must start with '---\\n'.")
    narr = s["narration"]
    if not isinstance(narr, list):
      raise RuntimeError(f"slide[{i}].narration must be array.")
    for j, line in enumerate(narr):
      if not isinstance(line, dict) or "text" not in line or "gestures" not in line:
        raise RuntimeError(f"slide[{i}].narration[{j}] must have text and gestures.")
      if not isinstance(line["text"], str) or not line["text"].strip():
        raise RuntimeError(f"slide[{i}].narration[{j}].text must be non-empty string.")
      gs = line["gestures"]
      if not isinstance(gs, list) or any(g not in ALLOWED_GESTURES for g in gs):
        raise RuntimeError(f"slide[{i}].narration[{j}].gestures must be among {ALLOWED_GESTURES}.")
      if "duration_sec" in line and not isinstance(line["duration_sec"], (int, float)):
        raise RuntimeError(f"slide[{i}].narration[{j}].duration_sec must be number if present.")
    manim = s["manim"]
    if not isinstance(manim, list):
      raise RuntimeError(f"slide[{i}].manim must be array.")
    for j, blk in enumerate(manim):
      if not isinstance(blk, dict):
        raise RuntimeError(f"slide[{i}].manim[{j}] must be object.")
      for req in ["id","description","code","overlay"]:
        if req not in blk:
          raise RuntimeError(f"slide[{i}].manim[{j}] missing key: {req}")
      ov = blk["overlay"]
      if not isinstance(ov, dict) or ov.get("position") not in ["top-left","top-right","bottom-left","bottom-right"]:
        raise RuntimeError(f"slide[{i}].manim[{j}].overlay.position invalid.")
      if "width" not in ov or not isinstance(ov["width"], str):
        raise RuntimeError(f"slide[{i}].manim[{j}].overlay.width must be string like '40%'.")
      if "start_at_sec" in ov and not isinstance(ov["start_at_sec"], (int, float)):
        raise RuntimeError(f"slide[{i}].manim[{j}].overlay.start_at_sec must be number if present.")

def _call_openai(messages: List[dict], model: str, max_retries: int, backoff: float) -> dict:
  client = _ensure_openai()
  last_exc = None
  for attempt in range(max_retries):
    try:
      kwargs = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
      }
      # temperature 지원 여부에 따라만 추가
      if not model.startswith("gpt-5-mini"):
        kwargs["temperature"] = 0.2
      resp = client.chat.completions.create(**kwargs)
      content = (resp.choices[0].message.content or "").strip()
      plan = _parse_json_lenient(content)
      return plan
    except Exception as e:
      last_exc = e
      msg = str(e).lower()
      if "rate limit" in msg or "rate_limit_exceeded" in msg or getattr(e, "status_code", None) == 429:
        _sleep_with_hint(e, attempt, backoff)
        continue
      raise
  raise last_exc  # type: ignore

def make_script(
    solutions_path: str,
    outdir: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 8,
    backoff: float = 1.6,
    verbose: bool = False,
) -> str:
  global VERBOSE
  VERBOSE = verbose

  # Set module globals for CLI arg access
  globals()["ARGS_PER_PROBLEM"] = False
  globals()["ARGS_MIN_SECONDS"] = 60
  globals()["ARGS_MAX_SECONDS"] = 600
  globals()["ARGS_SECONDS_PER_1K"] = 120

  data = _read_json(solutions_path)
  chapter = data.get("chapter") or "?"
  problems: Dict[str, dict] = data.get("problems", {})
  _log(f"Loaded {len(problems)} solved problems for chapter {chapter}")
  _log(f"Mode: per-problem; duration targeting: min={getattr(sys.modules[__name__], 'ARGS_MIN_SECONDS', 60)}s, max={getattr(sys.modules[__name__], 'ARGS_MAX_SECONDS', 600)}s, scale={getattr(sys.modules[__name__], 'ARGS_SECONDS_PER_1K', 120)}/1k chars")

  plans = []
  for idx, (k, meta) in enumerate(problems.items(), start=1):
    tgt = _estimate_target_seconds(
      meta.get("solution_markdown",""),
      min_s=getattr(sys.modules[__name__], "ARGS_MIN_SECONDS", 60),
      max_s=getattr(sys.modules[__name__], "ARGS_MAX_SECONDS", 600),
      per_1k=getattr(sys.modules[__name__], "ARGS_SECONDS_PER_1K", 120)
    )
    _log(f"[{idx}/{len(problems)}] Building plan for problem {k} (target ~{tgt}s)...")
    messages = _build_prompt_for_problem(chapter, k, meta, tgt)
    plan_piece = _call_openai(messages, model=model, max_retries=max_retries, backoff=backoff)
    # enforce current problem on slides and validate
    for s in plan_piece.get("slides", []) or []:
      if s.get("problem") in (None, "", "null"):
        s["problem"] = k
    if plan_piece.get("chapter") != chapter:
      plan_piece["chapter"] = chapter
    for s in plan_piece.get("slides", []):
      if isinstance(s.get("marp_md"), str) and not s["marp_md"].startswith("---\n"):
        s["marp_md"] = "---\n" + s["marp_md"].lstrip()
    try:
      _validate_plan(plan_piece)
    except RuntimeError as e:
      print(traceback.format_exc())
    plans.append(plan_piece)
  plan = _merge_plans(plans, chapter)
  if plan.get("schema_version") != SCHEMA_VERSION:
    plan["schema_version"] = SCHEMA_VERSION
  _log(f"Parsed plan with {len(plan.get('slides', []) or [])} slides.")

  # Assemble MARP
  marp_md = _assemble_marp(plan)

  out_json = {
    "chapter": chapter,
    "plan": plan,
  }

  out_json_path = os.path.join(outdir, f"script-{chapter}.json")
  out_marp_path = os.path.join(outdir, f"slides-{chapter}.md")
  _write_json(out_json_path, out_json)
  _write_text(out_marp_path, marp_md)

  # Emit Manim code files
  manim_paths = _emit_manim_blocks(plan, outdir)
  _log(f"Wrote script JSON to {out_json_path}, MARP deck to {out_marp_path}, Manim files: {len(manim_paths)}")
  return out_marp_path

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Turn solver output into a lecture-style script.")
  p.add_argument("--solutions", required=True, help="Path to solver JSON (e.g., out/solutions/solutions-1.2.json)")
  p.add_argument("--outdir", required=True, help="Directory to write script JSON/MD")
  p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
  p.add_argument("--max-retries", type=int, default=8)
  p.add_argument("--backoff", type=float, default=1.6)
  p.add_argument("--verbose", action="store_true")
  p.add_argument("--min-seconds", type=int, default=60, help="Minimum target narration duration per problem (seconds).")
  p.add_argument("--max-seconds", type=int, default=600, help="Maximum target narration duration per problem (seconds).")
  p.add_argument("--seconds-per-1kchars", type=int, default=120, help="Scale seconds by solution length: target ~= min(max(min_seconds, len_chars/1000 * seconds_per_1kchars), max_seconds).")
  return p.parse_args(argv)

if __name__ == "__main__":
  args = _parse_args()
  # Set module globals for helpers
  ARGS_MIN_SECONDS = args.min_seconds
  ARGS_MAX_SECONDS = args.max_seconds
  ARGS_SECONDS_PER_1K = args.seconds_per_1kchars
  try:
    make_script(
      solutions_path=args.solutions,
      outdir=args.outdir,
      model=args.model,
      max_retries=args.max_retries,
      backoff=args.backoff,
      verbose=args.verbose,
    )
  except Exception as e:
    print(f"[SCRIPTER ERROR] {e}", file=sys.stderr)
    sys.exit(1)
