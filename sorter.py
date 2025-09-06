"""
sorter.py

If there is an instruction image that applies to multiple problems (e.g., "boyce-1.1-instruction-1_2_3_4.png"), this sorter will automatically attach that instruction image to each of the listed problems when building the problem bundles. The instruction images will be sorted before the problem’s own pages (using page = -1). This way, when uploading to the API, each problem’s image list will start with the relevant instruction if one exists.

Groups textbook problems (provided as images) into bundles that must be solved
together because a problem explicitly depends on an earlier problem.

Workflow
--------
1) Scan an images directory (e.g., problems/images) and cluster pages that
   belong to the same problem number (e.g., "boyce 1.2-5-1.png", "boyce 1.2-5-2.png").
2) Ask an OpenAI vision model to read each problem statement and detect
   dependencies such as:
     - "Follow the instructions for Problem 1"
     - "Use the method of Problem 5"
     - "Also see Problem 21 of Section 1.1"
3) If every dependency is within the provided set, union the problems into
   bundles (e.g., {1,2}, {5,6}). If a dependency references a *previous
   section* or any problem not in the set, raise an error indicating which
   earlier problem is required (the caller can fetch that problem).
4) Emit a JSON file describing problems, dependencies, bundles, and a default
   video order (textbook order). This JSON is the hand‑off to the scripter/
   solver stages.

CLI
---
  python -m sorter --images-dir problems/images --output problems/bundles.json

Requirements
------------
- OPENAI_API_KEY must be set in the environment.
- Uses OpenAI Python SDK v1 (pip install openai>=1.30).
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional, Set
import time
import random

# Optional import guard: we only import the OpenAI client when we actually call the API,
# so local unit tests for the parser can run without the package present.
try:
  from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
  OpenAI = None  # type: ignore


# Compile filename/instruction regexes for a given book prefix.
from typing import Tuple
def _compile_patterns(book: str) -> Tuple[re.Pattern, re.Pattern]:
  """
  Compile filename patterns for a given book prefix. The book name itself
  should not contain spaces or underscores. Filenames may use either a hyphen
  or a single space between the book and the chapter (e.g., "boyce-1.2-5.png"
  or "boyce 1.2-5.png"). The book prefix is optional.
  """
  book_pat = re.escape(book)
  prefix = rf"(?i)(?:^|/)(?:{book_pat}[-\s]+)?"
  filename_regex = re.compile(
      rf"{prefix}(?P<chap1>\d+)\.(?P<chap2>\d+)-(?P<pnum>\d+)(?:-(?P<page>\d+))?\.(?:png|jpg|jpeg|pdf)$"
  )
  instruction_regex = re.compile(
      rf"{prefix}(?P<chap1>\d+)\.(?P<chap2>\d+)-instruction-(?P<plist>\d+(?:_\d+)*)\.(?:png|jpg|jpeg|pdf)$"
  )
  return filename_regex, instruction_regex

# Valid external dependency identifier: e.g., "1.1-21"
EXTERNAL_ID_RE = re.compile(r"^\d+\.\d+-\d+$")


@dataclasses.dataclass(frozen=True)
class ProblemImage:
  chapter: str           # e.g., "1.2"
  problem: int           # e.g., 5
  page: Optional[int]    # e.g., 1 or None
  path: str


def _natural_key(s: str) -> Tuple:
  """Sort helper that keeps numeric portions in numeric order."""
  return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))


def scan_images(
    images_dir: str,
    chapter_filter: Optional[str] = None,
    book: str = "boyce"
) -> Dict[int, List[ProblemImage]]:
  """
  Discover problem images in `images_dir` and group them by problem number.
  Returns { problem_number: [ProblemImage, ... sorted by page/path] }.

  Accepts file names like:
    - mybook 1.2-5.png
    - mybook 1.2-5-1.png, mybook 1.2-5-2.png
    - mybook-1.1-instruction-1_2_3_4.png  (applies to problems 1,2,3,4 in chapter 1.1)

  The `book` parameter sets the book prefix to match in filenames (default: "boyce").
  The book name itself should not contain spaces or underscores. Filenames may use
  either a hyphen or a single space after the book prefix, and the prefix is optional.

  If `chapter_filter` is provided, only images whose chapter matches the filter
  string (e.g., "1.2") will be included.
  """
  if not os.path.isdir(images_dir):
    raise FileNotFoundError(f"Images directory not found: {images_dir}")

  filename_re, instruction_re = _compile_patterns(book)

  grouped: Dict[int, List[ProblemImage]] = defaultdict(list)
  instructions: Dict[int, List[ProblemImage]] = defaultdict(list)
  chapter_seen: Optional[str] = None

  for root, _, files in os.walk(images_dir):
    for fname in sorted(files, key=_natural_key):
      mi = instruction_re.search(os.path.join(root, fname))
      if mi:
        chap = f"{mi.group('chap1')}.{mi.group('chap2')}"
        if chapter_filter is not None and chap != chapter_filter:
          continue
        plist = [int(x) for x in mi.group('plist').split('_') if x]
        # Create a ProblemImage with page=-1 so it sorts before normal pages.
        pi_base = ProblemImage(chapter=chap, problem=-1, page=-1, path=os.path.join(root, fname))
        for pnum in plist:
          instructions[pnum].append(dataclasses.replace(pi_base, problem=pnum))
        if chapter_seen is None:
          chapter_seen = chap
        continue

      m = filename_re.search(os.path.join(root, fname))
      if not m:
        continue
      chap = f"{m.group('chap1')}.{m.group('chap2')}"
      if chapter_filter is not None and chap != chapter_filter:
        continue
      if chapter_seen is None:
        chapter_seen = chap
      problem = int(m.group('pnum'))
      page = int(m.group('page')) if m.group('page') else None
      grouped[problem].append(ProblemImage(chapter=chap, problem=problem, page=page, path=os.path.join(root, fname)))

  for pnum, imgs in instructions.items():
    grouped[pnum].extend(imgs)

  # Sort pages per problem (by page then path)
  for pnum in grouped:
    grouped[pnum].sort(key=lambda im: (im.page if im.page is not None else 0, _natural_key(im.path)))

  if not grouped:
    raise RuntimeError(f"No valid problem images found under: {images_dir}")

  return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def _encode_image_to_data_uri(path: str) -> str:
  mime = "image/png"
  lower = path.lower()
  if lower.endswith(".jpg") or lower.endswith(".jpeg"):
    mime = "image/jpeg"
  elif lower.endswith(".pdf"):
    mime = "application/pdf"
  with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
  return f"data:{mime};base64,{b64}"


def _build_messages_for_dependencies(problems: Dict[int, List[ProblemImage]]) -> list:
  """
  Construct the multimodal message payload for the OpenAI Chat Completions API.
  We ask the model to return STRICT JSON with two maps:

    {
      "dependencies": { "6": [5], "2": [1] },
      "external_dependencies": { "9": "1.1-21" }
    }

  - `dependencies[i]` lists earlier problems (by number) within the provided set that
    problem i explicitly references and thus cannot be solved independently.
  - `external_dependencies[i]` uses a string identifier when a problem points to an earlier
    section or to a problem not included in this batch (e.g., "1.1-21").
  """
  # System prompt keeps the model scoped and forces JSON-only output.
  system_text = (
    "You are a careful teaching assistant. You will be given images of several textbook problems "
    "from the SAME section. Identify any problem that cannot be solved independently because it "
    "explicitly instructs to consult or use the method of an EARLIER problem. Detect phrases like "
    "\"Follow the instructions for Problem k\", \"Use the method of Problem k\", \"Also see Problem k\", "
    "or similar. Only consider dependencies on EARLIER problems.\n\n"
    "Return STRICT JSON with two keys:\n"
    "{\n"
    "  \"dependencies\": { \"i\": [k1, k2, ...] },\n"
    "  \"external_dependencies\": { \"i\": \"section-problem\" }\n"
    "}\n"
    "- Use integers as strings for keys in the JSON so it is valid.\n"
    "- In `dependencies`, list only problem numbers that are present in THIS BATCH and are EARLIER.\n"
    "- In `external_dependencies`, include entries that point to a problem OUTSIDE this batch or prior section (e.g., \"1.1-21\").\n"
    "- If none, return empty objects for those keys."
  )

  user_chunks = [
    {
      "type": "text",
      "text": (
        "Here are the problems. For each, read the statement carefully and detect if it references "
        "an EARLIER problem in this batch or a problem in a previous section. Then output JSON as specified."
      ),
    }
  ]
  for pnum, images in problems.items():
    user_chunks.append({"type": "text", "text": f"Problem {pnum}:"})
    for im in images:
      user_chunks.append(
        {"type": "image_url", "image_url": {"url": _encode_image_to_data_uri(im.path)}}
      )

  return [
    {"role": "system", "content": [{"type": "text", "text": system_text}]},
    {"role": "user", "content": user_chunks},
  ]


def _parse_json_block(s: str) -> dict:
  """
  Best-effort JSON extraction: try to parse the whole string as JSON,
  otherwise extract the first {...} block heuristically.
  """
  s = s.strip()
  try:
    return json.loads(s)
  except Exception:
    pass

  # Heuristic: find the first JSON object block.
  m = re.search(r"\{[\s\S]*\}", s)
  if not m:
    raise ValueError(f"Model did not return JSON: {s[:200]}")
  return json.loads(m.group(0))


def ask_dependencies_with_openai(
    problems: Dict[int, List[ProblemImage]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_retries: int = 6,
    backoff_base: float = 1.5,
) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
  """
  Call OpenAI to detect dependency relations.

  Returns:
    deps: { problem -> [earlier problem numbers within the batch] }
    extern: { problem -> 'section-problem' }  # dependencies outside this batch
  """
  if OpenAI is None:
    raise RuntimeError(
      "openai package is not installed. Run `pip install openai>=1.30`."
    )
  client = OpenAI()

  messages = _build_messages_for_dependencies(problems)
  last_exc = None
  for attempt in range(max_retries):
    try:
      resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
      )
      text = resp.choices[0].message.content or ""
      data = _parse_json_block(text)
      raw_deps = data.get("dependencies", {}) or {}
      raw_ext = data.get("external_dependencies", {}) or {}
      # Normalize to int-keyed dicts
      deps: Dict[int, List[int]] = {
        int(k): sorted({int(x) for x in v if str(x).isdigit()})
        for k, v in raw_deps.items()
      }
      # Normalize and filter externals: strip values, keep only valid external ids
      _extern0 = {int(k): (str(v).strip() if v is not None else "") for k, v in raw_ext.items()}
      extern: Dict[int, str] = {k: v for k, v in _extern0.items() if EXTERNAL_ID_RE.match(v)}
      _log(f"Normalized externals: kept {len(extern)}/{len(_extern0)} (filtered empties/invalid).")
      return deps, extern
    except Exception as e:
      last_exc = e
      # Try to detect rate limit errors (status_code 429 or error message)
      msg = str(e).lower()
      status_code = getattr(e, "status_code", None)
      is_rate_limit = (
        (status_code == 429)
        or ("rate limit" in msg)
        or ("rate_limit_exceeded" in msg)
      )
      if is_rate_limit and attempt < max_retries - 1:
        # Try to parse suggested wait time from error message, e.g. "3.765s"
        m = re.search(r"([0-9]*\.?[0-9]+)s", str(e))
        if m:
          sleep = float(m.group(1))
        else:
          sleep = backoff_base ** attempt + random.uniform(0, 0.5)
        time.sleep(sleep)
        continue
      else:
        raise
  # If we get here, all retries failed
  raise last_exc


class ExternalDependencyError(RuntimeError):
  pass


def build_bundles(
    problems: Dict[int, List[ProblemImage]],
    deps: Dict[int, List[int]],
) -> List[List[int]]:
  """
  Build bundles using union-find over the given dependencies.

  Example:
    deps = {2:[1], 6:[5]}  -> bundles [[1,2], [5,6], [3], [4], [7], ...]
  """
  # Union-Find (Disjoint Set)
  parent: Dict[int, int] = {p: p for p in problems.keys()}

  def find(x: int) -> int:
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def union(a: int, b: int) -> None:
    ra, rb = find(a), find(b)
    if ra != rb:
      # attach smaller root to larger to keep trees shallow
      if ra < rb:
        parent[rb] = ra
      else:
        parent[ra] = rb

  for p, reqs in deps.items():
    for q in reqs:
      if q in problems and p in problems:
        union(p, q)

  groups: Dict[int, List[int]] = defaultdict(list)
  for p in problems.keys():
    groups[find(p)].append(p)

  # Sort each group and return in textbook order by min element
  bundles = [sorted(v) for v in groups.values()]
  bundles.sort(key=lambda lst: min(lst))
  return bundles


def make_output_json(
    images: Dict[int, List[ProblemImage]],
    bundles: List[List[int]],
    deps: Dict[int, List[int]],
    extern: Dict[int, str],
    chapter: Optional[str] = None,
    paired_bundles: Optional[List[List[str]]] = None,
    lecture_order: Optional[List[int]] = None,
    external_problem_images: Optional[Dict[str, List[str]]] = None,
) -> dict:
  problems_list = []
  for pnum, plist in images.items():
    problems_list.append(
      {
        "number": pnum,
        "images": [im.path for im in plist],
      }
    )
  problems_list.sort(key=lambda x: x["number"])

  out = {
    "problems": problems_list,
    "dependencies": {str(k): v for k, v in sorted(deps.items())},
    "bundles": bundles,
    "external_dependencies": {str(k): v for k, v in sorted(extern.items())},
    "video_order": [p["number"] for p in problems_list],
  }
  if chapter is not None:
    out["chapter"] = chapter
  if paired_bundles is not None:
    out["paired_bundles"] = paired_bundles
  if lecture_order is not None:
    out["lecture_order"] = lecture_order
  if external_problem_images is not None:
    out["external_problem_images"] = external_problem_images
  return out


def _parse_external_identifier(s: str) -> Tuple[str, int]:
  """
  Parse strings like "1.1-21" into ("1.1", 21).
  """
  parts = s.split("-")
  if len(parts) != 2:
    raise ValueError(f"Invalid external dependency format: {s}")
  chapter_str = parts[0]
  try:
    prob_num = int(parts[1])
  except Exception:
    raise ValueError(f"Invalid problem number in external dependency: {s}")
  return chapter_str, prob_num


def run_sorter(
    images_dir: str,
    output_path: Optional[str] = None,
    model: str = "gpt-4o-mini",
    fail_on_external: bool = True,
    chapter: Optional[str] = None,
    max_retries: int = 6,
    backoff: float = 1.5,
    book: str = "boyce",
) -> dict:
  """
  End-to-end: scan, query OpenAI, build bundles, and optionally save JSON.

  If `fail_on_external` is True and the batch references an earlier problem
  outside the set (e.g., another section), raise ExternalDependencyError with
  a helpful message naming the missing prerequisite(s).
  """
  global VERBOSE
  # Log book prefix if verbose
  if "VERBOSE" in globals() and VERBOSE:
    _log(f"Using book prefix: {book}")
  elif not "VERBOSE" in globals():
    # Support for VERBOSE in some environments
    VERBOSE = False
  if VERBOSE:
    _log(f"Using book prefix: {book}")

  primary_images = scan_images(images_dir, chapter_filter=chapter, book=book)
  all_images = scan_images(images_dir, chapter_filter=None, book=book)

  deps, extern = ask_dependencies_with_openai(primary_images, model=model, max_retries=max_retries, backoff_base=backoff)

  _log(f"Primary externals to resolve: {len(extern)}")

  # Resolve external dependencies that reference earlier problems in other chapters
  resolved_external_pairs: List[List[str]] = []
  external_problem_images: Dict[str, List[str]] = {}

  unresolved_externals = {}

  for pnum, ext_id in extern.items():
    try:
      chap_str, prob_num = _parse_external_identifier(ext_id)
    except ValueError:
      # Keep as unresolved if format is unexpected
      unresolved_externals[pnum] = ext_id
      continue
    # Check if the referenced problem exists in all_images
    if prob_num in all_images:
      # Verify chapter matches the external id
      # The chapter of the external problem images is all_images[prob_num][0].chapter
      # but there could be multiple images, so just check the first one if exists
      if all_images[prob_num] and all_images[prob_num][0].chapter == chap_str:
        # Record pairing primary problem -> external id string
        resolved_external_pairs.append([f"{chapter}-{pnum}" if chapter is not None else str(pnum), ext_id])
        # Collect external problem images
        external_problem_images[ext_id] = [im.path for im in all_images[prob_num]]
      else:
        unresolved_externals[pnum] = ext_id
    else:
      unresolved_externals[pnum] = ext_id

  if fail_on_external and unresolved_externals:
    missing = ", ".join([f"{k}→{v}" for k, v in sorted(unresolved_externals.items())]) or "(none)"
    raise ExternalDependencyError(
      "Some problems require earlier problems from outside this batch. "
      f"Missing prerequisites: {missing}"
    )

  bundles = build_bundles(primary_images, deps)

  # lecture_order: only primary problems in textbook order (no external)
  lecture_order = sorted(primary_images.keys())

  out = make_output_json(
    primary_images,
    bundles,
    deps,
    extern,
    chapter=chapter,
    paired_bundles=resolved_external_pairs,
    lecture_order=lecture_order,
    external_problem_images=external_problem_images if external_problem_images else None,
  )

  if output_path:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(out, f, ensure_ascii=False, indent=2)

  return out


def _log(msg: str) -> None:
  print(f"[sorter] {msg}", file=sys.stderr)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Group problems into dependency bundles.")
  p.add_argument("--images-dir", required=True, help="Directory containing problem images.")
  p.add_argument("--output", help="Where to write the bundles JSON (optional).")
  p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use.")
  p.add_argument("--allow-external", action="store_true",
                 help="Do not fail when dependencies reference problems outside this batch; include them in the JSON instead.")
  p.add_argument("--chapter", type=str, default=None,
                 help="Only include problems from this chapter (e.g., '1.2').")
  p.add_argument("--book", type=str, default="boyce",
                 help="Book prefix used in filenames (e.g., 'boyce'). No spaces/underscores in the name. Filenames may use either '-' or ' ' after the prefix.")
  p.add_argument("--max-retries", type=int, default=6,
                 help="Maximum number of retries for OpenAI API calls (default: 6).")
  p.add_argument("--backoff", type=float, default=1.5,
                 help="Base for exponential backoff when retrying OpenAI API calls (default: 1.5).")
  return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
  args = _parse_args(argv)
  try:
    out = run_sorter(
      images_dir=args.images_dir,
      output_path=args.output,
      model=args.model,
      fail_on_external=not args.allow_external,
      chapter=args.chapter,
      max_retries=args.max_retries,
      backoff=args.backoff,
      book=args.book,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
  except ExternalDependencyError as e:
    # Fail with a clear message so upstream orchestrator can fetch the needed earlier problem.
    print(f"[SORTER ERROR] {e}", file=sys.stderr)
    sys.exit(2)
  except Exception as e:
    print(f"[SORTER ERROR] {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
  main()
