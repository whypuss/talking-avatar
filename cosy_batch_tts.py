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
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model

COSY_MODEL = "jackuh02/CosyVoice2-Yue-ZoengJyutGaai-mlx"


def split_sentences(text: str, max_len: int = 38) -> list[str]:
    """Split Traditional Chinese script into natural short sentences for stable TTS."""
    clean = re.sub(r"\s+", "", text.strip())
    if not clean:
        return []
    raw = [p.strip() for p in re.split(r"(?<=[。！？!?；;])", clean) if p.strip()]
    out: list[str] = []
    for sent in raw:
        if len(sent) <= max_len:
            out.append(sent)
            continue
        # Further split long sentences at commas while keeping punctuation
        clauses = [c for c in re.split(r"(?<=[，,、])", sent) if c]
        buf = ""
        for c in clauses:
            if buf and len(buf) + len(c) > max_len:
                out.append(buf)
                buf = c
            else:
                buf += c
        if buf:
            out.append(buf)
    return out


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


def build_subtitle_chunks(sentence: str, start_sec: float, end_sec: float, max_chars: int = 22) -> list[dict]:
    """Break a sentence into readable subtitle lines proportioned over its exact audio span."""
    raw_parts = [p.strip() for p in re.split(r"[，。！？；、,\.\n]+", sentence) if p.strip()]
    lines: list[str] = []
    for part in raw_parts:
        while len(part) > max_chars:
            lines.append(part[:max_chars])
            part = part[max_chars:]
        if part:
            lines.append(part)
    if not lines:
        return []
    total_chars = sum(max(1, len(x)) for x in lines)
    dur = max(0.05, end_sec - start_sec)
    cursor = start_sec
    entries = []
    for i, line in enumerate(lines):
        if i == len(lines) - 1:
            seg_end = end_sec
        else:
            seg_end = cursor + dur * (len(line) / total_chars)
        entries.append({"start": round(cursor, 3), "end": round(seg_end, 3), "text": line})
        cursor = seg_end
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch CosyVoice2-Yue TTS with exact subtitle timeline.")
    parser.add_argument("--text", required=True, help="Traditional Chinese text to synthesize.")
    parser.add_argument("--ref-audio", required=True, help="Reference audio WAV path.")
    parser.add_argument("--ref-text", required=True, help="Exact transcript of reference audio.")
    parser.add_argument("--output-wav", required=True, help="Output 24kHz WAV path.")
    parser.add_argument("--output-timeline", help="Optional output JSON timeline path for synced subtitles.")
    parser.add_argument("--model", default=COSY_MODEL, help="MLX CosyVoice2 model repo/path.")
    parser.add_argument("--pause-sec", type=float, default=0.22, help="Pause duration between sentences.")
    parser.add_argument("--ref-pad-sec", type=float, default=0.30, help="Trailing silence padded to ref audio.")
    parser.add_argument("--lead-trim-sec", type=float, default=0.0, help="Fixed leading trim per segment (sec).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier.")
    args = parser.parse_args()

    sentences = split_sentences(args.text)
    if not sentences:
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
        pause_samples = np.zeros(int(sr * args.pause_sec), dtype=np.float32)

        for idx, sent in enumerate(sentences):
            prefix = str(tmp_dir / f"seg_{idx:02d}")
            print(f"[CosyVoice2-Yue] ({idx + 1}/{len(sentences)}) {sent}")
            generate_audio(
                text=sent,
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
                pause_samples = np.zeros(int(sr * args.pause_sec), dtype=np.float32)

            cleaned = trim_and_clean_segment(data, sr, lead_trim_sec=args.lead_trim_sec)
            seg_dur = len(cleaned) / float(sr)
            timeline.extend(build_subtitle_chunks(sent, cur_sec, cur_sec + seg_dur))
            all_audio.append(cleaned)
            cur_sec += seg_dur

            if idx < len(sentences) - 1 and len(pause_samples) > 0:
                all_audio.append(pause_samples)
                cur_sec += args.pause_sec

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
