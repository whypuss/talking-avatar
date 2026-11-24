#!/usr/bin/env python3
"""
End-to-end Image-to-Talking-Avatar pipeline (whytalkv3 Streaming + Apple MPS accelerated).
Turns any portrait image + text (or audio) into a full-frame HD talking-head MP4:
  1. High-sample-rate original audio preservation + 16kHz analysis track
  2. Single-pass InsightFace lite detection (detection + 3D 68-pt landmarks once, zero per-frame overhead)
  3. Continuous news-anchor head motion, slow natural blinks & micro-saccades (zero chunk jump-cuts)
  4. Streaming batch Wav2Lip-GAN directly on Apple MPS / CUDA / CPU (zero RAM blowup, zero DIVX re-encoding)
  5. Optional MPS GFPGAN v1.4 HD restoration (skips silent frames) OR fast LAB color-matched mouth sharpening
  6. Cached RGBA Traditional Chinese subtitle overlay (exact JSON timeline or proportional fallback)
  7. Single-pass FFmpeg pipe encode preserving original 24kHz+ audio with lip-sync offset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from insightface.app import FaceAnalysis


def run_cmd(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_audio(
    out_wav_16k: str,
    audio_path: str | None = None,
    text: str | None = None,
    text_file: str | None = None,
    voice: str = "Sinji",
    rate: int = 185,
) -> tuple[float, str]:
    """
    Create 16kHz mono PCM WAV for Wav2Lip analysis and return (duration_sec, mux_audio_path).
    When --audio is provided, preserves the original high-sample-rate audio for final MP4 muxing.
    """
    mux_audio = out_wav_16k
    if audio_path:
        mux_audio = audio_path
        run_cmd([
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav_16k,
        ])
    else:
        if text_file:
            text = Path(text_file).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Provide either --audio, --text, or --text-file.")

        if voice.startswith("edge:"):
            edge_voice = voice.split(":", 1)[1]
            tmp_mp3 = out_wav_16k + ".mp3"
            run_cmd(["edge-tts", "--voice", edge_voice, "--text", text, "--write-media", tmp_mp3])
            run_cmd([
                "ffmpeg", "-y", "-i", tmp_mp3,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav_16k,
            ])
            os.remove(tmp_mp3)
        else:
            tmp_aiff = out_wav_16k + ".aiff"
            run_cmd(["say", "-v", voice, "-r", str(rate), "-o", tmp_aiff, text])
            run_cmd([
                "ffmpeg", "-y", "-i", tmp_aiff,
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav_16k,
            ])
            os.remove(tmp_aiff)

    data, sr = sf.read(out_wav_16k)
    return len(data) / float(sr), mux_audio


def detect_face_once(img: np.ndarray, insightface_root: str) -> dict:
    """
    Run InsightFace buffalo_l (lite modules only: detection + 3D 68 landmarks) ONCE on the portrait.
    Returns bbox, eye_boxes, jaw_pts, pivot, and soft head_mask, then immediately frees the ONNX models.
    """
    h, w = img.shape[:2]
    app = FaceAnalysis(
        name="buffalo_l",
        root=insightface_root,
        allowed_modules=["detection", "landmark_3d_68"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = app.get(img)
    if not faces:
        app.prepare(ctx_id=-1, det_size=(320, 320))
        faces = app.get(img)
    if not faces:
        raise ValueError("No face detected in the input portrait image.")

    face = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )[0]

    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    fw, fh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    pivot = (cx, min(float(h - 1), y2 + fh * 0.34))

    # Soft elliptical head+hair mask so only the head sways smoothly while studio background stays locked
    yy, xx = np.ogrid[:h, :w]
    rx, ry = fw * 1.04, fh * 1.22
    dist = ((xx - cx) / max(1.0, rx)) ** 2 + ((yy - (cy - fh * 0.06)) / max(1.0, ry)) ** 2
    mask = np.clip(1.40 - dist, 0.0, 1.0).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(7.0, fw * 0.10))
    head_mask = mask[..., None]

    # Precompute eye ROI boxes and soft elliptical masks once
    eye_items: list[tuple[int, int, int, int, np.ndarray]] = []
    if hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 2:
        ew = max(12, int(fw * 0.22))
        eh = max(8, int(fh * 0.11))
        for ex, ey in face.kps[:2]:
            ex_i, ey_i = int(round(ex)), int(round(ey))
            ex1 = max(0, ex_i - ew // 2)
            ey1 = max(0, ey_i - eh // 2)
            ex2 = min(w, ex_i + ew // 2)
            ey2 = min(h, ey_i + eh // 2)
            rh, rw = ey2 - ey1, ex2 - ex1
            if rh >= 4 and rw >= 4:
                ey_y, ey_x = np.ogrid[:rh, :rw]
                emask = np.clip(
                    1.2 - (((ey_x - rw / 2.0) / (rw * 0.48)) ** 2 + ((ey_y - rh / 2.0) / (rh * 0.45)) ** 2),
                    0.0,
                    1.0,
                ).astype(np.float32)
                emask = cv2.GaussianBlur(emask, (0, 0), sigmaX=max(1.5, rw * 0.08))[..., None]
                eye_items.append((ex1, ey1, ex2, ey2, emask))

    jaw_pts = None
    if getattr(face, "landmark_3d_68", None) is not None:
        pts68 = face.landmark_3d_68[:, :2]
        jaw_pts = np.vstack([pts68[3:14], pts68[35:30:-1]]).astype(np.float32)

    del app
    return {
        "bbox": (x1, y1, x2, y2),
        "fw": fw,
        "fh": fh,
        "pivot": pivot,
        "head_mask": head_mask,
        "eye_items": eye_items,
        "jaw_pts": jaw_pts,
    }


def compute_motion_trajectory(
    audio_16k: str,
    total_frames: int,
    fps: int,
    amp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Precompute calm, natural broadcast-anchor head motion and organic blink schedule.

    Design goals:
      - Low-frequency, restrained nod cadence (~6-9/min, never bobbing every second)
      - Organic, non-periodic subtle head tilt with critically damped 2nd-order smoothing
      - Gentle breathing + ultra-slow posture sway so quiet stretches stay alive, never robotic
      - Rare, gentle glance-down only at natural speech pauses in longer broadcasts
    """
    rms_env = np.ones(total_frames, dtype=np.float32)
    if audio_16k and os.path.exists(audio_16k):
        try:
            y, sr = sf.read(audio_16k)
            if y.ndim > 1:
                y = y.mean(axis=1)
            hop = max(1, int(sr / fps))
            rms = np.array([
                math.sqrt(float(np.mean(y[i * hop : min((i + 1) * hop, len(y))] ** 2)))
                if i * hop < len(y) else 0.0
                for i in range(total_frames)
            ], dtype=np.float32)
            p95 = float(np.percentile(rms, 95)) if len(rms) > 0 else 1.0
            rms_env = np.clip(rms / (p95 + 1e-6), 0.0, 1.2)
        except Exception:
            pass

    dt = 1.0 / float(max(1, fps))
    rng = np.random.default_rng(20261124)

    dx_arr = np.zeros(total_frames, dtype=np.float32)
    dy_arr = np.zeros(total_frames, dtype=np.float32)
    ang_arr = np.zeros(total_frames, dtype=np.float32)

    # 2nd-order critically damped spring states for ultra-smooth, jerk-free head movement
    cur_ang, vel_ang, tgt_ang = 0.0, 0.0, 0.0
    cur_dy, vel_dy = 0.0, 0.0
    omega_ang = 1.55   # ~2.0s smooth settle time for subtle tilt
    omega_dy = 4.2     # ~0.75s smooth settle time for gentle nods

    # Organic timers (in frames)
    nod_timer, nod_dur_frames = 0, int(fps * 0.80)
    nod_amp_px = 0.0
    nod_cooldown = int(fps * float(rng.uniform(2.0, 3.2)))
    silent_run = 0

    tilt_hold = 0
    tilt_cooldown = int(fps * float(rng.uniform(4.0, 6.5)))
    tilt_dir = 1.0 if float(rng.random()) > 0.5 else -1.0

    glance_timer, glance_dur_frames = 0, int(fps * 1.15)
    glance_cooldown = int(fps * float(rng.uniform(22.0, 30.0)))

    for idx in range(total_frames):
        t = idx * dt
        e = float(rms_env[idx])
        is_speaking = e > 0.14

        if is_speaking:
            came_from_pause = silent_run >= int(fps * 0.18)
            silent_run = 0
        else:
            came_from_pause = False
            silent_run += 1

        # 1. Restrained micro-nods: only on phrase restarts or rare strong emphasis, every ~5-8s
        if nod_cooldown > 0:
            nod_cooldown -= 1
        elif is_speaking and nod_timer == 0 and glance_timer == 0:
            if came_from_pause or (e > 0.80 and idx > int(fps * 2.0)):
                nod_dur_frames = int(fps * float(rng.uniform(0.76, 1.00)))
                nod_timer = nod_dur_frames
                nod_amp_px = float(rng.uniform(1.15, 1.75)) * amp
                nod_cooldown = int(fps * float(rng.uniform(5.2, 8.2)))

        # 2. Subtle non-periodic head tilt: every ~8-14s, small amplitude (<= 1.0 deg), sometimes returns to neutral
        if tilt_hold > 0:
            tilt_hold -= 1
            if tilt_hold == 0:
                tgt_ang = 0.0
                tilt_cooldown = int(fps * float(rng.uniform(6.5, 11.5)))
        elif tilt_cooldown > 0:
            tilt_cooldown -= 1
        elif is_speaking:
            # 25% chance to stay centered a bit longer instead of alternating like a metronome
            if float(rng.random()) < 0.25:
                tgt_ang = 0.0
                tilt_cooldown = int(fps * float(rng.uniform(4.5, 7.5)))
            else:
                mag = min(1.05, float(rng.uniform(0.80, 1.35)) * amp)
                tgt_ang = tilt_dir * mag
                tilt_dir *= -1.0 if float(rng.random()) < 0.78 else 1.0
                tilt_hold = int(fps * float(rng.uniform(3.2, 5.2)))

        # 3. Rare soft glance-down at natural sentence pauses (only in broadcasts > 25s)
        if glance_cooldown > 0:
            glance_cooldown -= 1
        elif (
            total_frames > int(fps * 25.0)
            and glance_timer == 0
            and nod_timer == 0
            and silent_run >= int(fps * 0.16)
            and idx < total_frames - int(fps * 3.0)
        ):
            glance_dur_frames = int(fps * float(rng.uniform(1.05, 1.35)))
            glance_timer = glance_dur_frames
            glance_cooldown = int(fps * float(rng.uniform(30.0, 42.0)))

        # Evaluate smooth sin^2 pulse envelopes (zero velocity at start & end -> zero jerk)
        nod_dy = 0.0
        if nod_timer > 0:
            p = 1.0 - (nod_timer / float(max(1, nod_dur_frames)))
            nod_dy = nod_amp_px * (math.sin(math.pi * p) ** 2)
            nod_timer -= 1

        glance_dy = 0.0
        if glance_timer > 0:
            gp = 1.0 - (glance_timer / float(max(1, glance_dur_frames)))
            glance_dy = (1.65 * amp) * (math.sin(math.pi * gp) ** 2)
            glance_timer -= 1

        # Continuous ultra-gentle physiological breathing & posture micro-drift
        breath_dy = 0.28 * amp * math.sin(2.0 * math.pi * t / 4.8)
        drift_dx = 0.45 * amp * (
            0.65 * math.sin(2.0 * math.pi * t / 13.7 + 0.4)
            + 0.35 * math.sin(2.0 * math.pi * t / 23.1 + 1.9)
        )
        drift_ang = 0.14 * amp * math.sin(2.0 * math.pi * t / 17.3 + 0.7)

        # Critically damped spring update for angle
        acc_ang = (omega_ang ** 2) * ((tgt_ang + drift_ang) - cur_ang) - 2.0 * omega_ang * vel_ang
        vel_ang += acc_ang * dt
        cur_ang += vel_ang * dt

        # Critically damped spring update for vertical motion
        tgt_dy = nod_dy + glance_dy + breath_dy
        acc_dy = (omega_dy ** 2) * (tgt_dy - cur_dy) - 2.0 * omega_dy * vel_dy
        vel_dy += acc_dy * dt
        cur_dy += vel_dy * dt

        dx_arr[idx] = drift_dx
        dy_arr[idx] = cur_dy
        ang_arr[idx] = cur_ang

    # Organic slow blink schedule (never at 0s, start around 3.2s, varied natural 4.8-7.4s intervals)
    duration = total_frames / float(fps)
    blink_times: list[float] = []
    t_next = 3.2
    while t_next < duration - 0.5:
        blink_times.append(round(t_next, 2))
        t_next += float(rng.uniform(4.8, 7.4))

    return dx_arr, dy_arr, ang_arr, blink_times


def render_base_frame(
    img: np.ndarray,
    img_f32: np.ndarray,
    face_info: dict,
    idx: int,
    fps: int,
    amp: float,
    dx: float,
    dy: float,
    ang: float,
    blink_times: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a single dynamic anchor frame and return (frame_bgr, affine_mat_2x3)."""
    h, w = img.shape[:2]
    if amp <= 0:
        return img.copy(), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    t = idx / float(fps)
    src_img = img
    eye_items = face_info["eye_items"]
    if eye_items:
        blink_amt = 0.0
        for bt in blink_times:
            dt = t - bt
            if -0.065 <= dt <= 0.105:
                blink_amt = max(0.0, min(1.0, 1.0 - abs(dt / (0.065 if dt < 0 else 0.105))))
                break
        # Very subtle, slow teleprompter gaze micro-drift (no high-frequency jitter)
        saccade_x = 0.30 * (
            0.60 * math.sin(2.0 * math.pi * 0.21 * t)
            + 0.40 * math.sin(2.0 * math.pi * 0.35 * t + 1.1)
        )
        if blink_amt > 0.02 or abs(saccade_x) > 0.06:
            src_img = img.copy()
            for ex1, ey1, ex2, ey2, emask in eye_items:
                roi = img[ey1:ey2, ex1:ex2]
                rh, rw = roi.shape[:2]
                sy = max(0.15, 1.0 - 0.85 * blink_amt)
                m_eye = np.float32([[1.0, 0.0, saccade_x], [0.0, sy, (1.0 - sy) * rh * 0.65]])
                warped_eye = cv2.warpAffine(
                    roi, m_eye, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                )
                src_img[ey1:ey2, ex1:ex2] = np.clip(
                    warped_eye.astype(np.float32) * emask + roi.astype(np.float32) * (1.0 - emask),
                    0,
                    255,
                ).astype(np.uint8)

    scale = 1.0 + 0.0009 * amp * math.sin(2.0 * math.pi * t / 4.8)
    mat = cv2.getRotationMatrix2D(face_info["pivot"], float(ang), scale)
    mat[0, 2] += float(dx)
    mat[1, 2] += float(dy)
    warped = cv2.warpAffine(src_img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    head_mask = face_info["head_mask"]
    if head_mask is not None:
        frame = np.clip(
            warped.astype(np.float32) * head_mask + img_f32 * (1.0 - head_mask),
            0,
            255,
        ).astype(np.uint8)
    else:
        frame = warped
    return frame, mat.astype(np.float32)


def compute_silence_weights(audio_path: str, total_frames: int, fps: int = 25) -> np.ndarray:
    """Compute smooth per-frame speaking weight (0.0 on sustained pauses, 1.0 during speech)."""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    spf = max(1, int(sr / fps))
    rms = np.array([
        np.sqrt(np.mean(audio[i * spf : min((i + 1) * spf, len(audio))] ** 2))
        if i * spf < len(audio) else 0.0
        for i in range(total_frames)
    ])

    is_silent = rms < 0.012
    weights = np.ones(total_frames, dtype=np.float32)
    run = 0
    for i in range(total_frames):
        if is_silent[i]:
            run += 1
        else:
            if run >= 3:
                weights[i - run : i] = 0.0
            run = 0
    if run >= 3:
        weights[total_frames - run : total_frames] = 0.0

    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    weights = np.convolve(weights, kernel, mode="same")
    return np.clip(weights, 0.0, 1.0)


def enhance_mouth_fast(pred_patch: np.ndarray, base_patch: np.ndarray) -> np.ndarray:
    """
    Fast (<2ms) LAB color-matching + subtle unsharp detail recovery for Wav2Lip 96x96 output.
    Eliminates dull/grayish lip tint mismatch and restores crisp lip/teeth edges without GFPGAN overhead.
    """
    if pred_patch.shape != base_patch.shape or pred_patch.size == 0:
        return pred_patch

    # 1. Gentle LAB mean alignment on lower half (mouth/chin zone) to match anchor skin lighting
    ph = pred_patch.shape[0]
    p_lab = cv2.cvtColor(pred_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    b_lab = cv2.cvtColor(base_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    lo = ph // 2
    for c in range(3):
        diff = float(np.mean(b_lab[lo:, :, c]) - np.mean(p_lab[lo:, :, c]))
        p_lab[:, :, c] = np.clip(p_lab[:, :, c] + diff * 0.55, 0.0, 255.0)
    matched = cv2.cvtColor(p_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # 2. Subtle unsharp mask for crisp teeth & lip contour
    blur = cv2.GaussianBlur(matched, (0, 0), sigmaX=1.15)
    sharp = cv2.addWeighted(matched, 1.50, blur, -0.50, 0)
    return sharp


def _split_subtitle_segments(text: str, max_chars: int = 24) -> list[str]:
    try:
        from cantonese_tn import split_subtitle_lines
        return split_subtitle_lines(text, max_chars=max_chars) or [text.strip()]
    except Exception:
        raw_parts = [p.strip() for p in re.split(r"(?:[，。！？；、\n]|(?<!\d)[,\.](?!\d))+", text) if p.strip()]
        segments: list[str] = []
        for part in raw_parts:
            while len(part) > max_chars:
                segments.append(part[:max_chars])
                part = part[max_chars:]
            if part:
                segments.append(part)
        return segments or [text.strip()]


def _load_tc_font(font_size: int):
    from PIL import ImageFont
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


class SubtitleRenderer:
    """Pre-renders each unique subtitle line once as a cropped BGR+alpha strip for ~15x faster compositing."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        total_frames: int,
        audio_delay_ms: int = 120,
        subtitle_text: str | None = None,
        timeline_json: str | None = None,
    ) -> None:
        self.w = width
        self.h = height
        self.fps = fps
        self.delay_sec = audio_delay_ms / 1000.0
        self.entries: list[tuple[float, float, str]] = []
        self.cache: dict[str, tuple[int, int, int, int, np.ndarray, np.ndarray]] = {}
        self.font = None

        if timeline_json and os.path.exists(timeline_json):
            data = json.loads(Path(timeline_json).read_text(encoding="utf-8"))
            for item in data:
                txt = str(item.get("text", "")).strip()
                if txt:
                    self.entries.append((float(item["start"]), float(item["end"]), txt))
        elif subtitle_text:
            segments = _split_subtitle_segments(subtitle_text)
            total_chars = sum(max(1, len(s)) for s in segments)
            total_sec = total_frames / float(fps)
            cursor = 0.0
            for idx, seg in enumerate(segments):
                if idx == len(segments) - 1:
                    end_sec = total_sec
                else:
                    end_sec = cursor + total_sec * (len(seg) / total_chars)
                self.entries.append((cursor, end_sec, seg))
                cursor = end_sec

        if self.entries:
            # Broadcast-proportioned Traditional Chinese subtitle size (~36-40px on 720p)
            font_size = max(22, min(42, int(width * 0.032)))
            self.font = _load_tc_font(font_size)

    def _get_overlay(self, text: str) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
        if text in self.cache:
            return self.cache[text]
        from PIL import Image, ImageDraw

        dummy = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=self.font, stroke_width=2)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad_x = max(16, int(self.w * 0.018))
        pad_y = max(9, int(self.h * 0.014))

        x = max(pad_x + 4, (self.w - tw) // 2)
        y = int(self.h * 0.915) - th

        bx1 = max(0, x - pad_x)
        by1 = max(0, y - pad_y)
        bx2 = min(self.w, x + tw + pad_x)
        by2 = min(self.h, y + th + pad_y)

        bw, bh = bx2 - bx1, by2 - by1
        strip = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(strip)
        sdraw.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=10, fill=(10, 14, 24, 175))
        sdraw.text(
            (x - bx1 - bbox[0], y - by1 - bbox[1]),
            text,
            font=self.font,
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 235),
        )
        rgba = np.array(strip)
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
        alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
        item = (bx1, by1, bx2, by2, bgr, alpha)
        self.cache[text] = item
        return item

    def draw(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        if not self.entries:
            return frame_bgr
        t_audio = (frame_idx / float(self.fps)) - self.delay_sec
        active_text = None
        for start_s, end_s, txt in self.entries:
            if start_s - 0.04 <= t_audio <= end_s + 0.06:
                active_text = txt
                break
        if not active_text:
            return frame_bgr

        bx1, by1, bx2, by2, bgr, alpha = self._get_overlay(active_text)
        roi = frame_bgr[by1:by2, bx1:bx2].astype(np.float32)
        blended = np.clip(bgr * alpha + roi * (1.0 - alpha), 0, 255).astype(np.uint8)
        frame_bgr[by1:by2, bx1:bx2] = blended
        return frame_bgr


def load_wav2lip_model(wav2lip_dir: str, device: str):
    if wav2lip_dir not in sys.path:
        sys.path.insert(0, wav2lip_dir)
    from models import Wav2Lip

    checkpoint_path = os.path.join(wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing Wav2Lip checkpoint: {checkpoint_path}")

    model = Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def compute_mel_chunks(wav2lip_dir: str, audio_16k: str, fps: int) -> list[np.ndarray]:
    if wav2lip_dir not in sys.path:
        sys.path.insert(0, wav2lip_dir)
    import audio as wav2lip_audio

    wav = wav2lip_audio.load_wav(audio_16k, 16000)
    mel = wav2lip_audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        wav = wav + np.random.normal(0, 1e-6, wav.shape)
        mel = wav2lip_audio.melspectrogram(wav)

    mel_step_size = 16
    mel_idx_multiplier = 80.0 / float(fps)
    mel_chunks: list[np.ndarray] = []
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    return mel_chunks


def stream_generate_video(
    image_path: str,
    audio_16k: str,
    mux_audio_path: str,
    output_mp4: str,
    wav2lip_dir: str,
    gfpgan_weights: str,
    insightface_root: str,
    fps: int = 25,
    breathing_amp: float = 0.6,
    pads: list[int] | None = None,
    gfpgan_weight: float = 0.0,
    audio_delay_ms: int = 120,
    device: str = "mps",
    batch_size: int = 32,
    subtitle_text: str | None = None,
    subtitle_timeline: str | None = None,
) -> None:
    """Single-pass streaming Wav2Lip-GAN + blend + subtitles + FFmpeg encode."""
    t_start = time.time()
    pads = pads or [0, 16, 8, 8]
    pady1, pady2, padx1, padx2 = pads

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    w_even, h_even = w - (w % 2), h - (h % 2)
    if (w_even, h_even) != (w, h):
        img = img[:h_even, :w_even]
        w, h = w_even, h_even
    img_f32 = img.astype(np.float32)

    print("=== [2/4] Detecting face & landmarks once (InsightFace lite) ===")
    face_info = detect_face_once(img, insightface_root)
    bx1_0, by1_0, bx2_0, by2_0 = face_info["bbox"]
    fw, fh = face_info["fw"], face_info["fh"]
    bbox_corners = np.array(
        [[bx1_0, by1_0], [bx2_0, by1_0], [bx2_0, by2_0], [bx1_0, by2_0]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    jaw_pts_base = face_info["jaw_pts"].reshape(-1, 1, 2) if face_info["jaw_pts"] is not None else None

    # Fixed expanded ROI for optional GFPGAN so GFPGAN only processes the face zone
    rx1 = max(0, int(bx1_0 - fw * 0.55))
    ry1 = max(0, int(by1_0 - fh * 0.35))
    rx2 = min(w, int(bx2_0 + fw * 0.55))
    ry2 = min(h, int(by2_0 + fh * 0.55))

    mel_chunks = compute_mel_chunks(wav2lip_dir, audio_16k, fps)
    total_frames = len(mel_chunks)
    print(f"Video: {w}x{h} @ {fps}fps | {total_frames} frames ({total_frames / fps:.2f}s) | device={device}")

    dx_arr, dy_arr, ang_arr, blink_times = compute_motion_trajectory(audio_16k, total_frames, fps, breathing_amp)
    speak_weights = compute_silence_weights(audio_16k, total_frames, fps)
    sub_renderer = SubtitleRenderer(
        width=w,
        height=h,
        fps=fps,
        total_frames=total_frames,
        audio_delay_ms=audio_delay_ms,
        subtitle_text=subtitle_text,
        timeline_json=subtitle_timeline,
    )

    print(f"=== [3/4] Loading Wav2Lip-GAN on {device} ===")
    w2l_model = load_wav2lip_model(wav2lip_dir, device)

    restorer = None
    if gfpgan_weight > 0:
        if not os.path.exists(gfpgan_weights):
            raise FileNotFoundError(f"Missing GFPGAN weights: {gfpgan_weights}")
        from gfpgan import GFPGANer

        print(f"Loading GFPGAN v1.4 on {device} (weight={gfpgan_weight})")
        restorer = GFPGANer(
            model_path=gfpgan_weights,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=torch.device(device),
        )

    # Launch single-pass FFmpeg encoder reading raw BGR frames from stdin and muxing original HD audio
    delay_sec = audio_delay_ms / 1000.0
    filter_complex = (
        f"[0:v]tpad=stop_mode=clone:stop_duration={delay_sec:.3f}[v];"
        f"[1:a]adelay={audio_delay_ms}|{audio_delay_ms}[a]"
    )
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", mux_audio_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        output_mp4,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    assert ffmpeg_proc.stdin is not None

    blur_k = max(15, int(fw * 0.09) | 1)
    blur_sigma = max(5.0, blur_k / 3.0)

    print("=== [4/4] Streaming Wav2Lip + Blend + Subtitles -> FFmpeg ===")
    try:
        for b_start in range(0, total_frames, batch_size):
            b_end = min(total_frames, b_start + batch_size)
            b_len = b_end - b_start

            base_frames: list[np.ndarray] = []
            coords_list: list[tuple[int, int, int, int]] = []
            jaw_polys: list[np.ndarray | None] = []
            face_crops_96: list[np.ndarray] = []

            for i in range(b_start, b_end):
                bf, mat = render_base_frame(
                    img, img_f32, face_info, i, fps, breathing_amp,
                    float(dx_arr[i]), float(dy_arr[i]), float(ang_arr[i]), blink_times,
                )
                base_frames.append(bf)

                # Transform face box via exact head affine matrix (zero S3FD jitter)
                c_t = cv2.transform(bbox_corners, mat).reshape(-1, 2)
                fx1 = max(0, int(math.floor(float(np.min(c_t[:, 0])))) - padx1)
                fy1 = max(0, int(math.floor(float(np.min(c_t[:, 1])))) - pady1)
                fx2 = min(w, int(math.ceil(float(np.max(c_t[:, 0])))) + padx2)
                fy2 = min(h, int(math.ceil(float(np.max(c_t[:, 1])))) + pady2)
                coords_list.append((fy1, fy2, fx1, fx2))

                crop_96 = cv2.resize(bf[fy1:fy2, fx1:fx2], (96, 96), interpolation=cv2.INTER_AREA)
                face_crops_96.append(crop_96)

                if jaw_pts_base is not None:
                    j_t = cv2.transform(jaw_pts_base, mat).reshape(-1, 2)
                    jaw_polys.append(cv2.convexHull(j_t.astype(np.int32)))
                else:
                    jaw_polys.append(None)

            # Batch Wav2Lip inference on MPS / CUDA / CPU
            img_arr = np.asarray(face_crops_96, dtype=np.float32)
            img_masked = img_arr.copy()
            img_masked[:, 48:, :, :] = 0.0
            img_in = np.concatenate((img_masked, img_arr), axis=3) / 255.0
            mel_arr = np.asarray(mel_chunks[b_start:b_end], dtype=np.float32)
            mel_in = np.reshape(mel_arr, [b_len, mel_arr.shape[1], mel_arr.shape[2], 1])

            t_img = torch.from_numpy(np.transpose(img_in, (0, 3, 1, 2))).to(device)
            t_mel = torch.from_numpy(np.transpose(mel_in, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = w2l_model(t_mel, t_img)

            pred_np = np.clip(pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0, 0, 255).astype(np.uint8)
            del t_img, t_mel, pred

            # Blend each frame & stream straight to FFmpeg
            for j in range(b_len):
                frame_idx = b_start + j
                bf = base_frames[j]
                fy1, fy2, fx1, fx2 = coords_list[j]
                w_speak = float(speak_weights[frame_idx])
                jaw_poly = jaw_polys[j]

                if w_speak <= 0.02:
                    # Pure silence frame: zero blend & skip GFPGAN entirely
                    final_frame = bf
                else:
                    p_up = cv2.resize(pred_np[j], (fx2 - fx1, fy2 - fy1), interpolation=cv2.INTER_CUBIC)
                    if restorer is not None:
                        work_full = bf.copy()
                        work_full[fy1:fy2, fx1:fx2] = p_up
                        roi_in = work_full[ry1:ry2, rx1:rx2]
                        _, _, roi_enh = restorer.enhance(
                            roi_in,
                            has_aligned=False,
                            only_center_face=True,
                            paste_back=True,
                            weight=gfpgan_weight,
                        )
                        work_full[ry1:ry2, rx1:rx2] = roi_enh
                        enh_patch = work_full[fy1:fy2, fx1:fx2]
                    else:
                        enh_patch = enhance_mouth_fast(p_up, bf[fy1:fy2, fx1:fx2])

                    final_frame = bf.copy()
                    if jaw_poly is not None:
                        # Localize convex hull mask inside face box for fast blending
                        local_poly = jaw_poly.copy()
                        local_poly[:, :, 0] -= fx1
                        local_poly[:, :, 1] -= fy1
                        ph, pw = fy2 - fy1, fx2 - fx1
                        mask = np.zeros((ph, pw), dtype=np.float32)
                        cv2.fillConvexPoly(mask, local_poly, 1.0)
                        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), blur_sigma)
                        mask_3c = (mask * w_speak)[:, :, None]
                        base_patch = bf[fy1:fy2, fx1:fx2].astype(np.float32)
                        blended = np.clip(enh_patch.astype(np.float32) * mask_3c + base_patch * (1.0 - mask_3c), 0, 255).astype(np.uint8)
                        final_frame[fy1:fy2, fx1:fx2] = blended
                    else:
                        final_frame[fy1:fy2, fx1:fx2] = enh_patch

                if sub_renderer.entries:
                    final_frame = sub_renderer.draw(final_frame, frame_idx)

                ffmpeg_proc.stdin.write(final_frame.tobytes())

            if b_end % (batch_size * 2) == 0 or b_end == total_frames:
                elapsed = time.time() - t_start
                print(f"[Stream] {b_end}/{total_frames} frames ({elapsed:.1f}s elapsed, {b_end / max(0.1, elapsed):.1f} fps)")

        ffmpeg_proc.stdin.close()
        ret = ffmpeg_proc.wait()
        if ret != 0:
            raise RuntimeError(f"FFmpeg encoding failed with exit code {ret}")
    finally:
        if ffmpeg_proc.poll() is None:
            ffmpeg_proc.kill()
        if device == "mps":
            torch.mps.empty_cache()

    total_time = time.time() - t_start
    print(f"Done in {total_time:.1f}s ({total_frames / max(0.1, total_time):.1f} fps) -> {output_mp4}")


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    default_insightface = os.environ.get(
        "INSIGHTFACE_ROOT",
        "/Volumes/Lexar/ComfyUI/models/insightface"
        if Path("/Volumes/Lexar/ComfyUI/models/insightface").exists()
        else str(Path.home() / ".insightface"),
    )

    parser = argparse.ArgumentParser(description="Generate HD talking-avatar video from any portrait image.")
    parser.add_argument("--image", required=True, help="Input portrait image path.")
    parser.add_argument("--text", help="Text to speak (uses TTS).")
    parser.add_argument("--text-file", help="Text file to speak (uses TTS).")
    parser.add_argument("--audio", help="Existing audio file (skips TTS).")
    parser.add_argument("--output", default="output/result.mp4", help="Output MP4 path.")
    parser.add_argument("--voice", default="Sinji", help="macOS `say` voice (e.g. Sinji, Tingting) or `edge:<voice>`.")
    parser.add_argument("--rate", type=int, default=185, help="Speech rate (words per minute) for macOS `say`.")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS.")
    parser.add_argument("--breathing-amp", type=float, default=0.6, help="Subtle breathing amplitude in pixels (0 to disable).")
    parser.add_argument("--gfpgan-weight", type=float, default=0.0, help="GFPGAN restoration blend weight (0 to use fast LAB+sharpen).")
    parser.add_argument("--pads", type=int, nargs=4, default=[0, 16, 8, 8], help="Wav2Lip face padding: top bottom left right.")
    parser.add_argument("--audio-delay-ms", type=int, default=120, help="Audio delay in ms for lip-sync alignment.")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto (mps/cuda/cpu), mps, cuda, or cpu.")
    parser.add_argument("--batch-size", type=int, default=32, help="Streaming Wav2Lip batch size (default: 32, 8GB RAM safe).")
    parser.add_argument("--wav2lip-dir", default=os.environ.get("WAV2LIP_DIR", str(repo_dir / "Wav2Lip")))
    parser.add_argument("--gfpgan-weights", default=os.environ.get("GFPGAN_WEIGHTS", str(repo_dir / "weights" / "GFPGANv1.4.pth")))
    parser.add_argument("--insightface-root", default=default_insightface)
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files.")
    parser.add_argument("--subtitles", action="store_true", help="Burn Traditional Chinese subtitles from --text / --text-file.")
    parser.add_argument("--subtitle-text", help="Explicit Traditional Chinese subtitle text to burn onto the video.")
    parser.add_argument("--subtitle-timeline", help="Path to JSON subtitle timeline generated by cosy_batch_tts.py.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="talking_avatar_")

    try:
        audio_16k = os.path.join(work_dir, "audio_16k.wav")
        print("=== [1/4] Preparing audio ===")
        duration, mux_audio = prepare_audio(
            audio_16k,
            audio_path=args.audio,
            text=args.text,
            text_file=args.text_file,
            voice=args.voice,
            rate=args.rate,
        )
        print(f"Audio duration: {duration:.2f}s (mux source: {mux_audio})")

        sub_txt = args.subtitle_text
        if not sub_txt and args.subtitles:
            if args.text:
                sub_txt = args.text
            elif args.text_file and os.path.exists(args.text_file):
                sub_txt = Path(args.text_file).read_text(encoding="utf-8").strip()

        device = resolve_device(args.device)
        stream_generate_video(
            image_path=args.image,
            audio_16k=audio_16k,
            mux_audio_path=mux_audio,
            output_mp4=args.output,
            wav2lip_dir=args.wav2lip_dir,
            gfpgan_weights=args.gfpgan_weights,
            insightface_root=args.insightface_root,
            fps=args.fps,
            breathing_amp=args.breathing_amp,
            pads=args.pads,
            gfpgan_weight=args.gfpgan_weight,
            audio_delay_ms=args.audio_delay_ms,
            device=device,
            batch_size=args.batch_size,
            subtitle_text=sub_txt,
            subtitle_timeline=args.subtitle_timeline,
        )
    finally:
        if args.keep_temp:
            print(f"Intermediate files kept at: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
