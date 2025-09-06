# -*- coding: utf-8 -*-
"""
Generate TTS audio for each narration in script-<chapter>.json, using edge-tts.
- Reads: out/scripts/scripts-<chapter>.json (schema with plan.slides[].narration[])
- Normalizes Korean math text via tts_normalize.normalize_text (rule-based)
- Synthesizes speech to out/speech/<chapter>/<slide_idx>_<narr_idx>.mp3
- Measures duration (seconds) using mutagen
- Writes two JSONs:
    1) out/speech/speech-map-<chapter>.json : list of items with
       {slide_idx, slide_id, text, normalized_text, gestures, speech_path, duration_sec}
    2) out/speech/gestures-<chapter>.json : flattened list of {gestures, duration_sec}
       If an item has multiple gestures, split duration equally across them.

Usage:
    python speech_generator.py --script out/scripts/scripts-1.2.json \
        --chapter 1.2 --outdir out/speech --voice ko-KR-SunHiNeural --rate "+0%" --volume "+0%" --verbose

Requirements:
    pip install edge-tts mutagen
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List
import math

import mutagen
from mutagen.mp3 import MP3  # type: ignore

from tts_normalize import normalize_text

try:
    import edge_tts  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("edge-tts is required. Install with: pip install edge-tts") from e


def sec(x: float) -> float:
    return round(float(x), 3)


def guess_chapter_from_filename(p: Path) -> str:
    m = re.search(r"(\d+\.\d+)", p.name)
    return m.group(1) if m else "unknown"


VOLUME_PERCENT_RE = re.compile(r"^[+-]\d+%$")
VOLUME_DB_RE = re.compile(r"^[+-]\d+dB$", re.IGNORECASE)

def normalize_volume_arg(volume: str) -> str:
    """edge-tts expects '+N%'. If user gives '+NdB', convert approximately to percent.
    Mapping uses gain = 20*log10(1 + p/100) ≈ dB  ->  p = (10**(dB/20) - 1)*100
    """
    v = volume.strip()
    if VOLUME_PERCENT_RE.match(v):
        return v
    m = VOLUME_DB_RE.match(v)
    if m:
        sign = 1 if v[0] == '+' else -1
        db = int(v[1:-2])  # strip sign and 'dB'
        # Convert signed dB to signed percent delta
        if sign >= 0:
            p = (10 ** (db / 20.0) - 1.0) * 100.0
        else:
            # For attenuation, approximate inverse mapping
            p = (1.0 - 10 ** (-db / 20.0)) * 100.0
            p = -abs(p)
        pct = int(round(p))
        # Clamp to edge-tts plausible range
        pct = max(-100, min(100, pct))
        return f"{pct:+d}%"
    # Fallback to neutral
    return "+0%"


async def synth_one(text: str, out_path: Path, voice: str, rate: str, volume: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate, volume=volume)
        await communicate.save(str(out_path))
    except Exception as e:
        # Write a tiny silent placeholder and re-raise to let duration step fail gracefully
        with open(out_path, 'wb') as f:
            pass
        raise


def get_duration_seconds(audio_path: Path) -> float:
    a = mutagen.File(str(audio_path))
    if a is None:
        raise RuntimeError(f"Cannot read audio metadata: {audio_path}")
    if hasattr(a, "info") and getattr(a.info, "length", None):
        return sec(a.info.length)
    if audio_path.suffix.lower() == ".mp3":
        return sec(MP3(str(audio_path)).info.length)
    raise RuntimeError(f"Unsupported audio for duration: {audio_path}")


async def main_async(args):
    script_path = Path(args.script)
    chapter = args.chapter or guess_chapter_from_filename(script_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    norm_volume = normalize_volume_arg(args.volume)

    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    plan = script.get("plan") or {}
    slides: List[Dict[str, Any]] = plan.get("slides", [])

    speech_items: List[Dict[str, Any]] = []
    tasks = []

    # Synthesize per narration
    for s_idx, slide in enumerate(slides, start=1):
        slide_id = slide.get("id")
        narrs: List[Dict[str, Any]] = slide.get("narration", [])
        for n_idx, narr in enumerate(narrs, start=1):
            orig = narr.get("text", "").strip()
            if not orig:
                continue
            norm = normalize_text(orig)
            gestures = narr.get("gestures", [])
            speech_filename = f"{chapter.replace('.', '_')}-s{s_idx:03d}-n{n_idx:02d}.mp3"
            speech_path = outdir / chapter / speech_filename

            tasks.append(synth_one(norm, speech_path, args.voice, args.rate, norm_volume))

            speech_items.append({
                "slide_idx": s_idx,
                "slide_id": slide_id,
                "text": orig,
                "normalized_text": norm,
                "gestures": gestures,
                "speech_path": str(speech_path.as_posix()),
                # duration will be filled after synthesis
                "duration_sec": None,
            })

    if args.verbose:
        print(f"[SPEECH] Synthesizing {len(tasks)} narration lines with edge-tts…")

    # Run TTS in parallel batches to avoid network throttling
    BATCH = args.batch
    for i in range(0, len(tasks), BATCH):
        await asyncio.gather(*tasks[i:i+BATCH])

    # Fill durations
    for item in speech_items:
        try:
            d = get_duration_seconds(Path(item["speech_path"]))
        except Exception:
            d = 0.0
        item["duration_sec"] = d

    # Write speech map JSON
    speech_map_path = outdir / f"speech-map-{chapter}.json"
    with open(speech_map_path, "w", encoding="utf-8") as f:
        json.dump(speech_items, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"[SPEECH] Wrote {speech_map_path}")

    # Build gestures-only flattened mapping
    gestures_only: List[Dict[str, Any]] = []
    for item in speech_items:
        gestures = item.get("gestures") or []
        dur = float(item.get("duration_sec") or 0.0)
        if not gestures:
            continue
        per = sec(dur / len(gestures)) if len(gestures) > 0 else 0.0
        for g in gestures:
            gestures_only.append({
                "gestures": g,
                "duration_sec": per,
            })

    gestures_path = outdir / f"gestures-{chapter}.json"
    with open(gestures_path, "w", encoding="utf-8") as f:
        json.dump(gestures_only, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"[SPEECH] Wrote {gestures_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate speech audio from script JSON using edge-tts")
    ap.add_argument("--script", required=True, help="Path to scripts-<chapter>.json")
    ap.add_argument("--chapter", required=False, help="Chapter label (e.g., 1.2)")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g., out/speech)")
    ap.add_argument("--voice", default="ko-KR-SunHiNeural", help="Edge TTS voice name")
    ap.add_argument("--rate", default="+0%", help="TTS rate, e.g. '+0%' or '-10%' ")
    ap.add_argument("--volume", default="+0%", help="TTS volume, e.g. '+0%' (edge-tts expects signed percent; dB input like '+3dB' is supported and converted)")
    ap.add_argument("--batch", type=int, default=6, help="Parallel synth batch size")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("[SPEECH] Interrupted")


if __name__ == "__main__":
    main()