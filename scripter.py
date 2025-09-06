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
    --model gpt-4o-mini \
    --max-retries 8 \
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
from typing import Dict, List, Optional

try:
  from openai import OpenAI  # type: ignore
except Exception:
  OpenAI = None  # type: ignore

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
      _write_text(fn, blk.get("code",""))
      paths.append(fn)
  return paths

def _build_prompt(chapter: str, problems: Dict[str, dict]) -> List[dict]:
  """
  Build a single combined prompt to convert per-problem solutions into a slide-centric plan.
  STRICT JSON schema required (see module docstring). Narration is attached to each slide.
  """
  sys_text = (
    "You are a lecture scriptwriter for an educational VTuber. Convert formal math solutions into a slide-centric "
    "lecture plan in Korean (존댓말) where each slide bundles: (1) MARP Markdown for the slide, "
    "(2) narration lines with gestures spoken while the slide is on screen, and (3) optional Manim blocks to overlay. "
    "Output STRICT JSON only, matching the schema in the instructions. No extra commentary, no Markdown, no code fences.\n\n"
    "Gesture tags (fixed set): ['point','emphasize','small-nod','smile','pause','write','think']\n\n"
    "Rules:\n"
    "- Use multiple equations per slide when needed to show step transitions; do NOT split every equation into a new slide.\n"
    "- Provide 2–4 slides per problem: statement → key derivations → final answer (with LaTeX \\boxed{...}).\n"
    "- For Manim blocks, include minimal runnable stubs and use MathTex to render any important formulas so equations are visible on screen.\n"
    "- Keep narration concise, natural Korean, referencing what is on the slide (avoid essay tone)."
  )

  # Flatten problems as text blocks (to inform the model).
  blocks: List[str] = []
  for key, meta in problems.items():
    sol = (meta.get('solution_markdown','') or '').strip()
    blocks.append(f"""### {key}
{sol}
""")

  user_text = (
    f"Chapter {chapter}. Here are the solved problems in order. "
    "Produce ONLY the JSON plan as per the slide-centric schema.\n\n" + "\n\n".join(blocks)
  )

  return [
    {"role": "system", "content": [{"type": "text", "text": sys_text}]},
    {"role": "user", "content": [{"type": "text", "text": user_text}]},
  ]

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

def _call_openai(messages: List[dict], model: str, max_retries: int, backoff: float) -> dict:
  client = _ensure_openai()
  last_exc = None
  for attempt in range(max_retries):
    try:
      resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
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

  data = _read_json(solutions_path)
  chapter = data.get("chapter") or "?"
  problems: Dict[str, dict] = data.get("problems", {})
  _log(f"Loaded {len(problems)} solved problems for chapter {chapter}")

  messages = _build_prompt(chapter, problems)
  _log(f"Requesting lecture plan from model for {len(problems)} problems...")
  plan = _call_openai(messages, model=model, max_retries=max_retries, backoff=backoff)
  # Basic validation
  if not isinstance(plan, dict) or "slides" not in plan:
    raise RuntimeError("Invalid plan JSON from model.")
  plan.setdefault("chapter", chapter)
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
  return p.parse_args(argv)

if __name__ == "__main__":
  args = _parse_args()
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
