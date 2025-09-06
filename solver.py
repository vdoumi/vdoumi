"""
solver.py
-----------
Reads sorter output (bundles JSON), calls an OpenAI vision+text model to
produce complete, step-by-step solutions per problem, while preserving
"session context" for problems that must be solved together (paired bundles
and in-batch dependencies). The result is a machine-consumable JSON that the
scripter can turn into a human-friendly lecture script.

Usage
-----
  python solver.py \
    --bundles problems/bundles-1.2.json \
    --outdir out/solutions \
    --model gpt-4o-mini \
    --max-retries 8 \
    --backoff 1.6 \
    --verbose

Input schema (produced by sorter)
---------------------------------
{
  "problems": [{"number": 5, "images": ["...png", ...]}, ...],
  "dependencies": {"6": [5], ...},
  "bundles": [[1,2], [5,6], ...],
  "external_dependencies": {"9": "1.1-21"},
  "paired_bundles": [["1.2-9","1.1-21"], ...],
  "lecture_order": [1,2,3,...],
  "chapter": "1.2",
  "external_problem_images": { "1.1-21": ["...png", ...] }
}

Output schema (per chapter)
---------------------------
{
  "chapter": "1.2",
  "model": "gpt-4o-mini",
  "problems": {
    "1.2-5": {
      "prompt": {...},           # sanitized prompt we sent
      "solution_markdown": "...",# model's step-by-step solution
      "images": ["...png"],
      "context_from": []         # e.g., ["1.2-5 (instruction)", "1.1-21 (reference)"]
    },
    ...
  }
}

Notes
-----
- We keep a "session context" per problem requiring earlier problems:
  - In-batch dependencies -> include the predecessor problem statements/solutions
    when prompting the successor (to ensure continuity).
  - Cross-chapter paired prerequisites -> include the external problem IMAGES as
    "reference context" (but do NOT produce a separate solution for them here).
- This file only solves problems. The natural lecture-style script is generated
  by scripter.py based on these solutions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
  from openai import OpenAI  # type: ignore
except Exception:
  OpenAI = None  # type: ignore

VERBOSE = False
def _log(msg: str) -> None:
  if VERBOSE:
    print(f"[SOLVER] {msg}", file=sys.stderr)

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

def _build_problem_index(bundles: dict) -> Dict[int, List[str]]:
  # { number -> image paths[] }
  idx: Dict[int, List[str]] = {}
  for p in bundles.get("problems", []):
    idx[int(p["number"])] = list(p["images"])
  return idx

def _ext_images_map(bundles: dict) -> Dict[str, List[str]]:
  return {k: list(v) for k, v in (bundles.get("external_problem_images") or {}).items()}

def _paired_map(bundles: dict) -> Dict[int, List[str]]:
  # "1.2-9","1.1-21" -> map 9 -> ["1.1-21"]
  mp: Dict[int, List[str]] = {}
  for pair in bundles.get("paired_bundles", []) or []:
    if not pair:
      continue
    primary = pair[0]  # e.g., "1.2-9"
    m = re.match(r"^(?P<chap>\d+\.\d+)-(?P<num>\d+)$", primary)
    if not m:
      continue
    num = int(m.group("num"))
    rest = [x for x in pair[1:]]
    if rest:
      mp.setdefault(num, []).extend(rest)
  return mp

def _images_for_problem(pnum: int, idx: Dict[int, List[str]]) -> List[str]:
  return idx.get(pnum, [])

def _messages_for_solution(
    chapter: str,
    pnum: int,
    images: List[str],
    predecessor_solutions: List[Tuple[str, str]],  # [(title, solution_md)]
    external_refs: List[Tuple[str, List[str]]],    # [(ext_id, images[])]
    short_mode: bool = False,
) -> List[dict]:
  """
  Assemble a multimodal prompt:
  - Images for the current problem first (instruction image, then pages)
  - Short textual instruction to produce a correct, fully explained solution
  - If predecessor solutions are given, include them as 'Earlier result' context
  - If external refs present, include their images as 'Reference (do not fully solve)'
  """
  sys_text = open("solver_prompt.txt").read()
  user_chunks: List[dict] = [{"type": "text", "text": f"Chapter {chapter}, Problem {pnum}. Solve completely."}]
  for path in images:
    user_chunks.append({"type": "image_url", "image_url": {"url": _encode_file(path)}})

  if predecessor_solutions:
    # Include summaries of predecessor results
    ctx = "\n\n".join([f"### Earlier Problem: {title}\n\n{sol}" for title, sol in predecessor_solutions])
    user_chunks.append({"type": "text", "text": f"Earlier results for reference (do not re-derive unless necessary):\n{ctx}"})

  if external_refs:
    user_chunks.append({"type": "text", "text": "Additional reference images from previous section (do not fully solve, only reuse method/idea):"})
    for ext_id, imgs in external_refs:
      user_chunks.append({"type": "text", "text": f"[Reference] {ext_id}"})
      for ip in imgs:
        user_chunks.append({"type": "image_url", "image_url": {"url": _encode_file(ip)}})

  extra = "Keep the solution within 800-1200 words." if short_mode else "No hard length limit; prioritize correctness."
  user_chunks.append({"type": "text", "text": extra})

  return [
    {"role": "system", "content": [{"type": "text", "text": sys_text}]},
    {"role": "user", "content": user_chunks},
  ]

def _encode_file(path: str) -> str:
  # For local files we pass as data URLs so the API can see them
  import base64, mimetypes
  mime = mimetypes.guess_type(path)[0] or "image/png"
  with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
  return f"data:{mime};base64,{b64}"

def _supports_temperature(model: str) -> bool:
  """
  Some models (e.g., 'gpt-5-mini') do not accept explicit temperature values,
  and only support the default. Return False for such models.
  """
  name = (model or "").lower()
  # Be conservative: models that explicitly say "only default temperature"
  # Avoid sending temperature for any 'gpt-5-mini' variants.
  return not ("gpt-5-mini" in name)

def _call_openai(messages: List[dict], model: str, max_retries: int, backoff: float, temperature: Optional[float]) -> str:
  client = _ensure_openai()
  last_exc = None
  for attempt in range(max_retries):
    try:
      kwargs = {"model": model, "messages": messages}
      if temperature is not None and _supports_temperature(model):
        kwargs["temperature"] = temperature
      resp = client.chat.completions.create(**kwargs)
      return (resp.choices[0].message.content or "").strip()
    except Exception as e:
      last_exc = e
      msg = str(e).lower()
      if "rate limit" in msg or "rate_limit_exceeded" in msg or getattr(e, "status_code", None) == 429:
        _sleep_with_hint(e, attempt, backoff)
        continue
      raise
  raise last_exc  # type: ignore

def solve_chapter(
    bundles_path: str,
    outdir: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 8,
    backoff: float = 1.6,
    temperature: Optional[float] = None,
    verbose: bool = False,
) -> str:
  global VERBOSE
  VERBOSE = verbose

  _log(f"Using model={model}, temperature={'default' if temperature is None else temperature}")

  data = _read_json(bundles_path)
  chapter = data.get("chapter") or "?"
  _log(f"Loaded bundles for chapter {chapter}")

  idx = _build_problem_index(data)
  deps: Dict[int, List[int]] = {int(k): list(v) for k, v in (data.get("dependencies") or {}).items()}
  paired = _paired_map(data)
  ext_map = _ext_images_map(data)
  lecture_order: List[int] = list(data.get("lecture_order") or sorted(idx.keys()))

  # We'll accumulate solutions so successors can see predecessors' results
  solutions: Dict[int, str] = {}

  out: Dict[str, dict] = {"chapter": chapter, "model": model, "problems": {}}

  for pnum in lecture_order:
    images = _images_for_problem(pnum, idx)
    if not images:
      _log(f"Skip {pnum}: no images")
      continue

    # Gather predecessor solutions (in-batch)
    preds = []
    for q in sorted(deps.get(pnum, [])):
      if q in solutions:
        preds.append((f"{chapter}-{q}", solutions[q]))

    # Gather external refs via paired mapping
    ext_refs: List[Tuple[str, List[str]]] = []
    for ext_id in paired.get(pnum, []):
      if ext_id in ext_map:
        ext_refs.append((ext_id, ext_map[ext_id]))

    messages = _messages_for_solution(chapter, pnum, images, preds, ext_refs)
    _log(f"Solving {chapter}-{pnum} with {len(images)} images, {len(preds)} predecessors, {len(ext_refs)} external refs")
    answer = _call_openai(messages, model=model, max_retries=max_retries, backoff=backoff, temperature=temperature)

    solutions[pnum] = answer
    key = f"{chapter}-{pnum}"
    out["problems"][key] = {
      "prompt": {"chapter": chapter, "problem": pnum, "images": images, "predecessors": [t for t, _ in preds], "external_refs": [e for e, _ in ext_refs]},
      "solution_markdown": answer,
      "images": images,
      "context_from": [t for t, _ in preds] + [f"{e} (reference)" for e, _ in ext_refs],
    }

  # Persist chapter solutions JSON
  out_path = os.path.join(outdir, f"solutions-{chapter}.json")
  _write_json(out_path, out)
  _log(f"Wrote solutions to {out_path}")
  return out_path

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Solve problems from sorter bundles JSON.")
  p.add_argument("--bundles", required=True, help="Path to sorter JSON (e.g., problems/bundles-1.2.json)")
  p.add_argument("--outdir", required=True, help="Directory to write solutions JSON")
  p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (vision-capable)")
  p.add_argument("--max-retries", type=int, default=8)
  p.add_argument("--backoff", type=float, default=1.6)
  p.add_argument("--temperature", type=float, default=None, help="Sampling temperature. If omitted, the default for the model is used. Some models (e.g., gpt-5-mini) do not allow setting this; we will skip sending it.")
  p.add_argument("--verbose", action="store_true")
  return p.parse_args(argv)

if __name__ == "__main__":
  args = _parse_args()
  try:
    solve_chapter(
      bundles_path=args.bundles,
      outdir=args.outdir,
      model=args.model,
      max_retries=args.max_retries,
      backoff=args.backoff,
      temperature=args.temperature,
      verbose=args.verbose,
    )
  except Exception as e:
    print(f"[SOLVER ERROR] {e}", file=sys.stderr)
    sys.exit(1)
