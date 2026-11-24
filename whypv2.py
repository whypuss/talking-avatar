#!/usr/bin/env python3
"""
whypv2: Dual-Image Cantonese News Broadcast Pipeline (Anchor + Scene B-roll + Sensitive Subtitle Censoring).

Built on top of the whyv2 HD talking-avatar engine:
  - Image 1 (--image): News anchor rendered with 1-shot MPS GFPGAN v1.4 HD mouth,
    portrait dual-region LAB lip color lock, text-driven Cantonese visemes, and calm 2nd-order head motion.
  - Image 2 (--image2 / --broll-image): News scene photo smoothly crossfaded during the on-scene
    segment (auto white-border trim, 16:9 cover crop, subtle Ken Burns zoom, corner badge, continuous subtitles).
  - Sensitive word censoring (--censor-subtitles enabled by default): masks high-risk words (e.g. 自殺/輕生/死亡)
    as 'xx' on burned-in subtitles while keeping spoken Cantonese TTS natural and intact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from whytalkv3 import (
    REPO_DIR,
    VOICE_PRESETS,
    WHY_PYTHON,
    get_duration,
    run_cmd,
    synthesize_cosyvoice_batch,
)


def auto_detect_broll_window(
    timeline_path: str | None,
    total_duration: float,
) -> tuple[float, float]:
    """
    Choose a natural broadcast cut-in window for Image 2 (B-roll) aligned to subtitle boundaries:
      - Lets the anchor open with the headline (~2.0s - 4.5s)
      - Cuts smoothly into the scene image for ~8s - 13s aligned to a sentence/clause boundary
      - Returns to the anchor for analysis/statistics and closing remarks.
    """
    if total_duration <= 6.0:
        return round(total_duration * 0.25, 2), round(total_duration * 0.75, 2)

    default_start = round(min(4.2, total_duration * 0.14), 2)
    default_end = round(min(total_duration - 2.0, default_start + min(11.5, total_duration * 0.28)), 2)

    if not timeline_path or not Path(timeline_path).exists():
        return default_start, default_end

    try:
        entries = json.loads(Path(timeline_path).read_text(encoding="utf-8"))
    except Exception:
        return default_start, default_end

    if not entries:
        return default_start, default_end

    # Pick start right after the opening headline clause (between 1.8s and 5.5s), or fallback
    b_start = default_start
    for item in entries:
        end_s = float(item.get("end", 0.0))
        if 1.8 <= end_s <= 5.5:
            b_start = round(end_s + 0.08, 2)
            break

    # Pick end at a natural pause/segment boundary giving 7.0s..14.0s of scene coverage
    b_end = default_end
    for item in entries:
        end_s = float(item.get("end", 0.0))
        span = end_s - b_start
        if 7.0 <= span <= 14.0 and end_s < total_duration - 2.0:
            b_end = round(end_s + 0.12, 2)
            # Prefer stopping right before transitional words like 此外 / 不過 / 至於
            idx = entries.index(item)
            if idx + 1 < len(entries):
                nxt_txt = str(entries[idx + 1].get("text", ""))
                if any(nxt_txt.startswith(k) for k in ("此外", "不過", "另外", "至於", "同時", "整體")):
                    break

    if b_end <= b_start + 1.5:
        b_end = round(min(total_duration - 1.5, b_start + min(10.0, total_duration * 0.28)), 2)
    return b_start, b_end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="whypv2: Dual-Image News Broadcast Pipeline (Anchor + Scene B-roll + Subtitle Censoring)"
    )
    parser.add_argument("--image", required=True, help="Image 1: Anchor portrait image path.")
    parser.add_argument(
        "--image2", "--broll-image", dest="broll_image",
        help="Image 2: Scene/news photo smoothly displayed during the broadcast.",
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
        help="Disable automatic sensitive-word subtitle censoring (censoring to 'xx' is ON by default in whypv2).",
    )
    parser.add_argument("--output", default=str(REPO_DIR / "output" / "whypv2_out.mp4"), help="Output MP4 path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.audio and not args.text:
        parser.error("Provide either --text or --audio.")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="whypv2_"))

    try:
        timeline_path: str | None = args.subtitle_timeline

        # 1. Audio & exact subtitle timeline
        if args.audio:
            audio_wav = args.audio
        else:
            preset = VOICE_PRESETS.get(args.voice_ref, {})
            ref_audio = args.ref_audio or preset.get("ref_audio")
            ref_text = args.ref_text or preset.get("ref_text")
            if not ref_audio or not ref_text:
                raise ValueError("Missing --ref-audio / --ref-text for voice cloning.")
            print(f"=== [whypv2 1/2] Synthesizing Cantonese voice ({args.voice_ref}) ===")
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
            print(f"=== [whypv2 2/2] Rendering HD anchor + B-roll ({b_start:.2f}s->{b_end:.2f}s) ===")
        else:
            print(f"=== [whypv2 2/2] Rendering HD anchor video ({duration:.1f}s) ===")

        run_cmd(cmd, cwd=str(REPO_DIR))
        print(f"Done! whypv2 output saved to: {out_path}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
