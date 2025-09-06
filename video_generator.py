# -*- coding: utf-8 -*-
"""
Compose a lecture video from slide PNGs + TTS speech + hard-burned subtitles (original text).

Inputs:
  --slides-dir out/slides               # files named slides.001.png, slides.002.png, ...
  --speech-map out/speech/speech-map-1.2.json  # produced by speech_generator.py
  --out out/video/lecture-1.2.mp4

Behavior:
  - For each entry in speech-map (ordered), pick the slide image at "slide_idx".
  - Show that slide for the speech duration, attach the audio, and hard-burn the original text at the bottom.
  - Concatenate all segments into a single MP4. (No VTuber overlay yet.)

Notes:
  - Uses MoviePy TextClip (requires ImageMagick for complex text). For Korean, pass a valid font via --font.
  - Subtitles are wrapped to a max width percentage and padded above the bottom.

Requirements:
    pip install moviepy
    # For better Korean font rendering ensure ImageMagick is available and a font like Noto Sans CJK is installed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
from moviepy.video.fx.resize import resize


def vprint(flag: bool, *msg):
    if flag:
        print("[VIDEO]", *msg)


def load_slides(slides_dir: Path) -> List[Path]:
    files = sorted(slides_dir.glob("slides.*.png"))
    if not files:
        raise SystemExit(f"No slide PNGs found under {slides_dir}. Expected names like 'slides.001.png'.")
    # Ensure files exist and are readable
    for p in files:
        if not p.exists():
            raise SystemExit(f"Slide path listed but not found: {p}")
    return files


def build_segment(slide_path: Path, text: str, audio_path: Path, duration: float,
                  size: tuple[int, int], font: str | None, fontsize: int, margin: int,
                  no_subtitles: bool, verbose: bool) -> CompositeVideoClip:
    if not slide_path.exists():
        raise SystemExit(f"Slide not found: {slide_path}")
    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")
    if duration <= 0:
        duration = 0.01

    W, H = size
    try:
        base = resize(ImageClip(str(slide_path), duration=duration), width=W, height=H)
    except Exception as e:
        raise SystemExit(f"Failed to open slide image: {slide_path} ({e})")

    try:
        audio = AudioFileClip(str(audio_path))
        base = base.set_audio(audio)
    except Exception as e:
        raise SystemExit(f"Failed to open audio: {audio_path} ({e})")

    if no_subtitles:
        return base.set_duration(duration)

    # Subtitle (may require ImageMagick)
    try:
        subtitle = TextClip(
            txt=text,
            fontsize=fontsize,
            font=font or "Arial",
            method="caption",
            size=(int(W*0.9), None),
            align="South",
        ).set_position(("center", H - margin - subtitle_h(fontsize)))
        clip = CompositeVideoClip([base, subtitle], size=(W, H))
    except Exception as e:
        vprint(verbose, f"Failed to render subtitles with TextClip (ImageMagick missing?). Continuing without subtitles. Detail: {e}")
        clip = base

    return clip.set_duration(duration)


def subtitle_h(fontsize: int) -> int:
    # Rough baseline for subtitle block height
    return int(fontsize * 3)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compose lecture video from slides + speech map")
    ap.add_argument("--slides-dir", required=True)
    ap.add_argument("--speech-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="1920x1080", help="Video resolution WxH, e.g., 1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--font", default=None, help="Font name for Korean subtitles (e.g., 'AppleGothic' or 'Noto Sans CJK KR')")
    ap.add_argument("--fontsize", type=int, default=40)
    ap.add_argument("--margin", type=int, default=40, help="Bottom margin for subtitles (px)")
    ap.add_argument("--no-subtitles", action="store_true", help="Disable hard-burned subtitles (for environments without ImageMagick)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    W, H = map(int, args.size.lower().split("x"))

    vprint(args.verbose, f"Slides dir: {args.slides_dir}, Speech map: {args.speech_map}, Out: {args.out}")

    slides = load_slides(Path(args.slides_dir))

    with open(args.speech_map, "r", encoding="utf-8") as f:
        speech_items: List[Dict[str, Any]] = json.load(f)

    # Ensure order by slide_idx then natural order within slide
    speech_items.sort(key=lambda x: (int(x["slide_idx"]), x["speech_path"]))

    clips = []
    for item in speech_items:
        idx = int(item["slide_idx"])  # 1-based
        try:
            slide_path = slides[idx - 1]
        except IndexError:
            raise SystemExit(f"Slide index {idx} has no matching PNG. Check slides in {args.slides_dir}")
        text = item["text"]
        audio_path = Path(item["speech_path"])  # stored as posix
        duration = float(item["duration_sec"]) or 0.01

        segment = build_segment(slide_path, text, audio_path, duration, (W, H), args.font, args.fontsize, args.margin, args.no_subtitles, args.verbose)
        clips.append(segment)

    video = concatenate_videoclips(clips, method="compose")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(str(out_path), fps=args.fps, codec="libx264", audio_codec="aac", threads=4)


if __name__ == "__main__":
    main()