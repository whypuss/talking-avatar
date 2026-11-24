#!/usr/bin/env python3
"""
whypv3: Complete Dual-Image Cantonese News Broadcast + Auto YouTube Cover Pipeline.

Extends whypv2 with automatic 1280x720 High-CTR YouTube thumbnail generation:
  1. Video Output (.mp4):
     - Image 1 (--image): whyv2 HD anchor (1-shot MPS GFPGAN v1.4 + portrait dual-region LAB lip color lock
       + Cantonese text-driven visemes + calm 2nd-order head motion + 24kHz CosyVoice2-Yue voice)
     - Image 2 (--image2): Auto-aligned news scene B-roll (white-border trim, Ken Burns push-in, smooth crossfade)
     - Sensitive subtitle censoring (--censor-subtitles ON by default: masks 自殺/輕生/死亡 as 'xx' on subtitles only)
  2. Cover Output (_cover.jpg, enabled by default):
     - Composites Image 1 (anchor + channel logo) + Image 2 (scene photo) with diagonal glowing breaking slash
       and auto-extracted (or manually overridden) YouTube-safe bright Traditional Chinese headline.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from whypv2 import auto_detect_broll_window
from whytalkv3 import (
    REPO_DIR,
    VOICE_PRESETS,
    WHY_PYTHON,
    get_duration,
    run_cmd,
    synthesize_cosyvoice_batch,
)
from yt_cover import generate_youtube_cover


def derive_cover_path(video_output: str, explicit_cover: str | None = None) -> Path:
    if explicit_cover:
        return Path(explicit_cover).resolve()
    vp = Path(video_output).resolve()
    return vp.with_name(f"{vp.stem}_cover.jpg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="whypv3: Dual-Image News Broadcast Video + Automatic 1280x720 YouTube Cover Generator"
    )
    parser.add_argument("--image", required=True, help="Image 1: Anchor portrait image path.")
    parser.add_argument(
        "--image2", "--broll-image", dest="broll_image",
        help="Image 2: News scene photo used for both mid-video B-roll and YouTube cover.",
    )
    parser.add_argument("--text", help="Traditional Chinese broadcast script.")
    parser.add_argument("--audio", help="Optional pre-generated audio WAV path (skips CosyVoice2-Yue).")
    parser.add_argument("--subtitle-text", help="Override subtitle text (defaults to --text).")
    parser.add_argument("--subtitle-timeline", help="Optional pre-generated JSON subtitle timeline.")
    parser.add_argument("--voice-ref", default="女主播h", help=f"Voice preset name: {list(VOICE_PRESETS.keys())}")
    parser.add_argument("--ref-audio", help="Custom reference audio WAV path.")
    parser.add_argument("--ref-text", help="Custom reference audio transcript.")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speech speed multiplier (default: 1.0).")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS (default: 25).")
    parser.add_argument("--breathing-amp", type=float, default=0.6, help="Head motion amplitude scale (default: 0.6).")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto (mps/cuda/cpu), mps, cuda, or cpu.")
    parser.add_argument("--batch-size", type=int, default=32, help="Streaming Wav2Lip batch size (default: 32).")
    parser.add_argument("--broll-start", type=float, help="Manual B-roll start time in seconds (auto if omitted).")
    parser.add_argument("--broll-end", type=float, help="Manual B-roll end time in seconds (auto if omitted).")
    parser.add_argument("--broll-label", default="現場資料", help="Subtle corner badge on B-roll (empty to hide).")
    parser.add_argument("--fast", action="store_true", help="Disable GFPGAN for ultra-fast preview.")
    parser.add_argument("--no-subtitles", action="store_true", help="Disable Traditional Chinese subtitles.")
    parser.add_argument(
        "--no-censor", action="store_true",
        help="Disable automatic sensitive-word subtitle censoring (ON by default in whypv3).",
    )
    parser.add_argument("--output", default=str(REPO_DIR / "output" / "whypv3_out.mp4"), help="Output MP4 path.")

    # YouTube cover options (enabled by default in whypv3)
    parser.add_argument("--no-cover", action="store_true", help="Skip generating the companion YouTube cover image.")
    parser.add_argument("--cover-only", action="store_true", help="Generate ONLY the YouTube cover image and exit.")
    parser.add_argument("--cover-output", help="Explicit cover JPEG path (defaults to <output_stem>_cover.jpg).")
    parser.add_argument("--cover-badge", help="Override top-right red badge on cover (e.g. 突發直擊).")
    parser.add_argument("--cover-loc", help="Override location tag on cover (e.g. 澳門青洲).")
    parser.add_argument("--cover-line1", help="Override main golden headline on cover.")
    parser.add_argument("--cover-line2-prefix", help="Override second-line prefix on cover.")
    parser.add_argument("--cover-highlight-val", help="Override highlighted number on cover (e.g. 39).")
    parser.add_argument("--cover-highlight-unit", help="Override highlighted unit on cover (e.g. 宗).")
    parser.add_argument("--cover-sub1", help="Override bottom summary strip left text.")
    parser.add_argument("--cover-sub2", help="Override bottom summary strip right text.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.cover_only and not args.audio and not args.text:
        parser.error("Provide either --text or --audio (or use --cover-only).")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cover_path: Path | None = None

    # Step 1 (or standalone): Generate companion 1280x720 YouTube cover
    if not args.no_cover:
        cover_path = derive_cover_path(str(out_path), args.cover_output)
        cover_script = args.text or args.subtitle_text or ""
        print(f"=== [whypv3 Cover] Generating 1280x720 YouTube cover -> {cover_path} ===")
        generate_youtube_cover(
            anchor_image=args.image,
            scene_image=args.broll_image,
            text=cover_script,
            output_path=str(cover_path),
            badge=args.cover_badge,
            location=args.cover_loc,
            line1=args.cover_line1,
            line2_prefix=args.cover_line2_prefix,
            highlight_val=args.cover_highlight_val,
            highlight_unit=args.cover_highlight_unit,
            sub1=args.cover_sub1,
            sub2=args.cover_sub2,
        )
        if args.cover_only:
            print(f"Done (cover-only)! Cover saved to: {cover_path}")
            return

    # Step 2: Generate dual-image HD Cantonese broadcast video (same core engine as whypv2)
    work_dir = Path(tempfile.mkdtemp(prefix="whypv3_"))
    try:
        timeline_path: str | None = args.subtitle_timeline

        if args.audio:
            audio_wav = args.audio
        else:
            preset = VOICE_PRESETS.get(args.voice_ref, {})
            ref_audio = args.ref_audio or preset.get("ref_audio")
            ref_text = args.ref_text or preset.get("ref_text")
            if not ref_audio or not ref_text:
                raise ValueError("Missing --ref-audio / --ref-text for voice cloning.")
            print(f"=== [whypv3 1/2] Synthesizing Cantonese voice ({args.voice_ref}) ===")
            tts_wav = work_dir / "tts_24k.wav"
            tts_tl = work_dir / "subtitles.json"
            audio_wav, timeline_path = synthesize_cosyvoice_batch(
                args.text, ref_audio, ref_text, tts_wav, tts_tl, speed=args.speed
            )

        duration = get_duration(audio_wav)
        gfpgan_w = "0" if args.fast else "0.60"
        sub_full = "" if args.no_subtitles else (args.subtitle_text or args.text or "")

        cmd = [
            str(WHY_PYTHON), str(REPO_DIR / "generate.py"),
            "--image", args.image,
            "--audio", audio_wav,
            "--fps", str(args.fps),
            "--breathing-amp", str(args.breathing_amp),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--gfpgan-weight", gfpgan_w,
            "--output", str(out_path),
        ]

        if not args.no_subtitles:
            if timeline_path and not args.subtitle_text:
                cmd.extend(["--subtitle-timeline", timeline_path])
            elif sub_full:
                cmd.extend(["--subtitle-text", sub_full])
            if not args.no_censor:
                cmd.append("--censor-subtitles")

        if args.broll_image:
            b_start, b_end = args.broll_start, args.broll_end
            if b_start is None or b_end is None:
                auto_s, auto_e = auto_detect_broll_window(timeline_path, duration)
                b_start = auto_s if b_start is None else b_start
                b_end = auto_e if b_end is None else b_end
            cmd.extend([
                "--broll-image", args.broll_image,
                "--broll-start", str(b_start),
                "--broll-end", str(b_end),
            ])
            if args.broll_label:
                cmd.extend(["--broll-label", args.broll_label])
            print(f"=== [whypv3 2/2] Rendering HD anchor + B-roll ({b_start:.2f}s->{b_end:.2f}s) ===")
        else:
            print(f"=== [whypv3 2/2] Rendering HD anchor video ({duration:.1f}s) ===")

        run_cmd(cmd, cwd=str(REPO_DIR))
        print(f"Done! whypv3 video saved to: {out_path}")
        if cover_path:
            print(f"Done! whypv3 YouTube cover saved to: {cover_path}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
