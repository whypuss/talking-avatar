#!/usr/bin/env python3
"""
whytalkv3: End-to-end Cantonese talking-avatar pipeline with:
  - Single-process sentence-split CosyVoice2-Yue voice cloning (anti-artifact ref padding + exact subtitle timeline)
  - Zero-chunk streaming Apple MPS Wav2Lip-GAN (eliminates 6s jump-cuts & 8GB OOM)
  - High-sample-rate 24kHz audio preservation + fast LAB mouth color matching & sharpening (or MPS GFPGAN --hd)
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
COSY_PYTHON = Path.home() / "projects" / "cosyvoice2-yue" / ".venv" / "bin" / "python3"
COSY_MODEL = "jackuh02/CosyVoice2-Yue-ZoengJyutGaai-mlx"
WHY_PYTHON = REPO_DIR / ".venv" / "bin" / "python"

_CLEAN_REF = REPO_DIR / "voices" / "女主播h_ref_clean.wav"
_FALLBACK_REF = REPO_DIR / "voices" / "女主播h_ref10s.wav"

VOICE_PRESETS = {
    "女主播h": {
        "ref_audio": str(_CLEAN_REF if _CLEAN_REF.exists() else _FALLBACK_REF),
        "ref_text": (
            "承認串謀分裂國家罪，分別判囚三年同三年兩個月。"
            if _CLEAN_REF.exists()
            else "承認串謀分裂國家罪，分別判囚三年同三年兩個月。本案涉及嘅港獨組"
        ),
    },
}


def run_cmd(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def get_duration(path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]).decode().strip()
    return float(out)


def split_text_for_chunks(text: str, num_chunks: int) -> list[str]:
    """Fallback proportional splitter only used when --chunk-sec > 0 is explicitly requested."""
    clean = text.strip()
    if num_chunks <= 1 or not clean:
        return [clean]
    parts = [p for p in re.split(r"(?<=[，。！？；、,])", clean) if p]
    if len(parts) >= num_chunks:
        buckets = [""] * num_chunks
        total_len = sum(len(p) for p in parts)
        target = total_len / num_chunks
        idx = 0
        for p in parts:
            if idx < num_chunks - 1 and len(buckets[idx]) >= target:
                idx += 1
            buckets[idx] += p
        return [b.strip() for b in buckets]
    step = math.ceil(len(clean) / num_chunks)
    return [clean[i * step : (i + 1) * step] for i in range(num_chunks)]


def synthesize_cosyvoice_batch(
    text: str,
    ref_audio: str,
    ref_text: str,
    out_wav: Path,
    out_timeline: Path,
    speed: float = 1.0,
) -> tuple[str, str]:
    if not COSY_PYTHON.exists():
        raise FileNotFoundError(f"CosyVoice2-Yue python not found: {COSY_PYTHON}")
    cmd = [
        str(COSY_PYTHON), str(REPO_DIR / "cosy_batch_tts.py"),
        "--text", text,
        "--ref-audio", ref_audio,
        "--ref-text", ref_text,
        "--model", COSY_MODEL,
        "--output-wav", str(out_wav),
        "--output-timeline", str(out_timeline),
        "--speed", str(speed),
    ]
    run_cmd(cmd, cwd=str(REPO_DIR))
    if not out_wav.exists():
        raise FileNotFoundError(f"Expected TTS output not found: {out_wav}")
    return str(out_wav), str(out_timeline)


def main() -> None:
    parser = argparse.ArgumentParser(description="whytalkv3: All-in-one Talking Avatar with Synced Subtitles")
    parser.add_argument("--image", required=True, help="Input portrait image path.")
    parser.add_argument("--text", help="Traditional Chinese text to speak and show as subtitles.")
    parser.add_argument("--audio", help="Optional pre-generated audio path (skips CosyVoice2-Yue).")
    parser.add_argument("--subtitle-text", help="Override subtitle text (defaults to --text).")
    parser.add_argument("--subtitle-timeline", help="Optional pre-generated JSON subtitle timeline.")
    parser.add_argument("--voice-ref", default="女主播h", help=f"Voice preset name: {list(VOICE_PRESETS.keys())}")
    parser.add_argument("--ref-audio", help="Custom reference audio WAV path.")
    parser.add_argument("--ref-text", help="Custom reference audio transcript.")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speech speed multiplier (default: 1.0).")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS (default: 25).")
    parser.add_argument("--breathing-amp", type=float, default=0.6, help="Head motion amplitude scale (0 to disable, default: 0.6).")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto (mps/cuda/cpu), mps, cuda, or cpu.")
    parser.add_argument("--batch-size", type=int, default=32, help="Streaming Wav2Lip batch size (default: 32).")
    parser.add_argument("--output", default=str(REPO_DIR / "output" / "whytalkv3_out.mp4"), help="Output MP4 path.")
    parser.add_argument(
        "--chunk-sec",
        type=int,
        default=0,
        help="Optional legacy hard-chunking in seconds (0 = seamless streaming single-pass, recommended).",
    )
    parser.add_argument("--hd", action="store_true", help="Enable MPS-accelerated GFPGAN v1.4 HD restoration.")
    parser.add_argument("--no-subtitles", action="store_true", help="Disable Traditional Chinese subtitles.")
    args = parser.parse_args()

    if not args.audio and not args.text:
        parser.error("Provide either --text or --audio.")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="whytalkv3_"))

    try:
        timeline_path: str | None = args.subtitle_timeline

        # 1. Audio preparation
        if args.audio:
            audio_wav = args.audio
        else:
            preset = VOICE_PRESETS.get(args.voice_ref, {})
            ref_audio = args.ref_audio or preset.get("ref_audio")
            ref_text = args.ref_text or preset.get("ref_text")
            if not ref_audio or not ref_text:
                raise ValueError("Missing --ref-audio / --ref-text for voice cloning.")
            print(f"=== [1/2] Synthesizing Cantonese voice ({args.voice_ref}) ===")
            tts_wav = work_dir / "tts_24k.wav"
            tts_tl = work_dir / "subtitles.json"
            audio_wav, timeline_path = synthesize_cosyvoice_batch(
                args.text, ref_audio, ref_text, tts_wav, tts_tl, speed=args.speed
            )

        duration = get_duration(audio_wav)
        gfpgan_w = "0.65" if args.hd else "0"
        sub_full = "" if args.no_subtitles else (args.subtitle_text or args.text or "")

        # 2. Video generation: seamless streaming single-pass by default (zero jump-cuts, low RAM)
        use_chunks = args.chunk_sec > 0 and duration > args.chunk_sec + 1.5
        if not use_chunks:
            print(f"=== [2/2] Generating seamless streaming video ({duration:.1f}s) ===")
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
            run_cmd(cmd, cwd=str(REPO_DIR))
        else:
            print(f"=== [2/3] Splitting {duration:.1f}s audio into {args.chunk_sec}s chunks ===")
            chunk_pattern = str(work_dir / "chunk_%02d.wav")
            run_cmd([
                "ffmpeg", "-y", "-i", audio_wav,
                "-f", "segment", "-segment_time", str(args.chunk_sec),
                "-c", "copy", chunk_pattern,
            ])
            chunks = sorted(work_dir.glob("chunk_*.wav"))
            sub_parts = split_text_for_chunks(sub_full, len(chunks)) if sub_full else [""] * len(chunks)
            out_videos: list[Path] = []

            for idx, chunk_wav in enumerate(chunks):
                print(f"--- Processing chunk {idx + 1}/{len(chunks)} ---")
                chunk_mp4 = work_dir / f"out_{idx:02d}.mp4"
                cmd = [
                    str(WHY_PYTHON), str(REPO_DIR / "generate.py"),
                    "--image", args.image,
                    "--audio", str(chunk_wav),
                    "--fps", str(args.fps),
                    "--device", args.device,
                    "--batch-size", str(args.batch_size),
                    "--gfpgan-weight", gfpgan_w,
                    "--output", str(chunk_mp4),
                ]
                if sub_parts[idx]:
                    cmd.extend(["--subtitle-text", sub_parts[idx]])
                run_cmd(cmd, cwd=str(REPO_DIR))
                out_videos.append(chunk_mp4)

            print("=== [3/3] Concatenating chunks ===")
            list_file = work_dir / "concat_list.txt"
            list_file.write_text("\n".join(f"file '{v}'" for v in out_videos), encoding="utf-8")
            run_cmd([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c", "copy", str(out_path),
            ])

        print(f"Done! whytalkv3 output saved to: {out_path}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
