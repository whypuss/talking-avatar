#!/usr/bin/env python3
"""
Single-process batch Cantonese TTS synthesizer using CosyVoice2-Yue-ZoengJyutGaai-mlx.
Loads the MLX model ONCE, pads reference audio trailing silence to prevent prefix artifacts,
splits long scripts into natural short sentences (avoiding long-context attention decay),
and outputs a clean 24kHz WAV plus an exact per-sentence subtitle timeline JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from cantonese_tn import (
    build_synced_subtitle_chunks,
    normalize_cantonese_tts,
    split_tts_segments,
)

COSY_MODEL = "jackuh02/CosyVoice2-Yue-ZoengJyutGaai-mlx"


def split_sentences(text: str, max_len: int = 44) -> list[str]:
    """Legacy helper returning display sentences from the balanced Cantonese splitter."""
    return [seg["display_text"] for seg in split_tts_segments(text, max_speech_len=max_len)]


def trim_and_clean_segment(
    audio: np.ndarray,
    sr: int,
    lead_trim_sec: float = 0.0,
    fade_ms: float = 15.0,
) -> np.ndarray:
    """Trim optional prefix artifact, strip excessive leading/trailing dead silence, apply short fades."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if lead_trim_sec > 0:
        cut = min(int(lead_trim_sec * sr), max(0, len(audio) - int(0.2 * sr)))
        audio = audio[cut:]

    # Trim dead silence at both ends while keeping a tiny natural margin
    win = max(1, int(sr * 0.02))
    rms = np.array([
        np.sqrt(np.mean(audio[i:i + win] ** 2))
        for i in range(0, len(audio), win)
    ])
    peak = float(np.max(rms)) if len(rms) > 0 else 0.0
    if peak > 1e-4:
        thresh = max(0.008, peak * 0.04)
        active = np.where(rms > thresh)[0]
        if len(active) > 0:
            start_idx = max(0, int(active[0]) * win - int(0.04 * sr))
            end_idx = min(len(audio), (int(active[-1]) + 1) * win + int(0.08 * sr))
            audio = audio[start_idx:end_idx]

    # Short fade-in/fade-out to prevent click/pop at joins
    fade_n = min(int(sr * fade_ms / 1000.0), len(audio) // 4)
    if fade_n > 1:
        audio[:fade_n] *= np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        audio[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

    return audio


def build_subtitle_chunks(sentence: str, start_sec: float, end_sec: float, max_chars: int = 24) -> list[dict]:
    """Break a display sentence into readable subtitle lines weighted by spoken Cantonese syllables."""
    return build_synced_subtitle_chunks(sentence, start_sec, end_sec, max_chars=max_chars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch CosyVoice2-Yue TTS with exact subtitle timeline.")
    parser.add_argument("--text", required=True, help="Traditional Chinese text to synthesize.")
    parser.add_argument("--ref-audio", required=True, help="Reference audio WAV path.")
    parser.add_argument("--ref-text", required=True, help="Exact transcript of reference audio.")
    parser.add_argument("--output-wav", required=True, help="Output 24kHz WAV path.")
    parser.add_argument("--output-timeline", help="Optional output JSON timeline path for synced subtitles.")
    parser.add_argument("--model", default=COSY_MODEL, help="MLX CosyVoice2 model repo/path.")
    parser.add_argument("--pause-sec", type=float, default=0.22, help="Pause duration between full sentences.")
    parser.add_argument("--clause-pause-sec", type=float, default=0.08, help="Short breath pause between split clauses.")
    parser.add_argument("--ref-pad-sec", type=float, default=0.30, help="Trailing silence padded to ref audio.")
    parser.add_argument("--lead-trim-sec", type=float, default=0.0, help="Fixed leading trim per segment (sec).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    args = parser.parse_args()

    segments = split_tts_segments(args.text)
    if not segments:
        raise ValueError("Empty input text.")

    with tempfile.TemporaryDirectory(prefix="cosy_batch_") as tmp:
        tmp_dir = Path(tmp)
        ref_pad = tmp_dir / "ref_pad.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", args.ref_audio, "-af", f"apad=pad_dur={args.ref_pad_sec:.2f}", str(ref_pad)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        print(f"[CosyVoice2-Yue] Loading model once: {args.model}")
        model = load_model(args.model)
        sr = int(getattr(model, "sample_rate", 24000))

        all_audio: list[np.ndarray] = []
        timeline: list[dict] = []
        cur_sec = 0.0

        for idx, seg in enumerate(segments):
            display_sent = seg["display_text"]
            tts_sent = seg["tts_text"]
            pause_kind = seg["pause_after"]
            prefix = str(tmp_dir / f"seg_{idx:02d}")
            if tts_sent != display_sent:
                print(f"[CosyVoice2-Yue] ({idx + 1}/{len(segments)}) {display_sent}  -->  [TTS] {tts_sent}")
            else:
                print(f"[CosyVoice2-Yue] ({idx + 1}/{len(segments)}) {display_sent}")

            generate_audio(
                text=tts_sent,
                model=model,
                ref_audio=str(ref_pad),
                ref_text=args.ref_text,
                file_prefix=prefix,
                audio_format="wav",
                join_audio=True,
                speed=args.speed,
                verbose=False,
            )
            seg_wav = Path(f"{prefix}.wav")
            if not seg_wav.exists():
                seg_wav = Path(f"{prefix}_000.wav")
            if not seg_wav.exists():
                raise FileNotFoundError(f"Generated segment WAV not found for prefix {prefix}")

            data, seg_sr = sf.read(str(seg_wav))
            if seg_sr != sr:
                sr = seg_sr

            cleaned = trim_and_clean_segment(data, sr, lead_trim_sec=args.lead_trim_sec)
            seg_dur = len(cleaned) / float(sr)
            timeline.extend(build_subtitle_chunks(display_sent, cur_sec, cur_sec + seg_dur))
            all_audio.append(cleaned)
            cur_sec += seg_dur

            if pause_kind == "sentence" and args.pause_sec > 0:
                all_audio.append(np.zeros(int(sr * args.pause_sec), dtype=np.float32))
                cur_sec += args.pause_sec
            elif pause_kind == "clause" and args.clause_pause_sec > 0:
                all_audio.append(np.zeros(int(sr * args.clause_pause_sec), dtype=np.float32))
                cur_sec += args.clause_pause_sec

        # Gentle peak normalization (-1 dBFS) for crisp broadcast loudness
        full = np.concatenate(all_audio, axis=0)
        peak = float(np.max(np.abs(full))) if len(full) > 0 else 0.0
        if peak > 1e-4:
            full = full * (0.89 / peak)

        out_wav = Path(args.output_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), full, sr, subtype="PCM_16")
        print(f"[CosyVoice2-Yue] Saved {cur_sec:.2f}s @ {sr}Hz -> {out_wav}")

        if args.output_timeline:
            out_tl = Path(args.output_timeline)
            out_tl.parent.mkdir(parents=True, exist_ok=True)
            out_tl.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[CosyVoice2-Yue] Saved subtitle timeline ({len(timeline)} entries) -> {out_tl}")


if __name__ == "__main__":
    main()
