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
    tight_mouth_pts = None
    pts68 = None
    mouth_ref_stats = None
    kps5 = face.kps.astype(np.float32) if getattr(face, "kps", None) is not None and len(face.kps) >= 5 else None

    if getattr(face, "landmark_3d_68", None) is not None:
        pts68 = face.landmark_3d_68[:, :2].astype(np.float32)
        jaw_pts = np.vstack([pts68[3:14], pts68[35:30:-1]]).astype(np.float32)

        # Build a tight mouth + mid-chin hull so original cheek/jawline skin pixels stay 100% untouched
        outer_lip = pts68[48:60]
        lip_c = outer_lip.mean(axis=0)
        lip_exp = lip_c + (outer_lip - lip_c) * np.array([1.34, 1.82], dtype=np.float32)
        chin_mid = pts68[6:11]
        tight_mouth_pts = np.vstack([lip_exp, chin_mid]).astype(np.float32)

        # Calibrate exact original lip & mouth-skin LAB statistics once from the source portrait
        inner_lip = pts68[60:68]
        m_out = np.zeros((h, w), dtype=np.uint8)
        m_in = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m_out, [outer_lip.astype(np.int32)], 255)
        cv2.fillPoly(m_in, [inner_lip.astype(np.int32)], 255)
        lip_ring = cv2.subtract(m_out, m_in)
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_ring = cv2.subtract(cv2.dilate(m_out, k_outer), cv2.dilate(m_out, k_inner))

        lab_ref = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lip_px = lab_ref[lip_ring > 0]
        skin_px = lab_ref[skin_ring > 0]
        if len(lip_px) >= 20 and len(skin_px) >= 40:
            l_lo, l_hi = np.percentile(lip_px[:, 0], [15, 92])
            a_lo = np.percentile(lip_px[:, 1], 30)
            valid = (lip_px[:, 0] >= l_lo) & (lip_px[:, 0] <= l_hi) & (lip_px[:, 1] >= a_lo)
            lip_clean = lip_px[valid] if int(valid.sum()) >= 20 else lip_px
            mouth_ref_stats = {
                "lip_mean": lip_clean.mean(axis=0).astype(np.float32),
                "lip_std": np.maximum(lip_clean.std(axis=0), 2.5).astype(np.float32),
                "skin_mean": skin_px.mean(axis=0).astype(np.float32),
                "skin_std": np.maximum(skin_px.std(axis=0), 2.5).astype(np.float32),
            }

    del app
    return {
        "bbox": (x1, y1, x2, y2),
        "fw": fw,
        "fh": fh,
        "pivot": pivot,
        "head_mask": head_mask,
        "eye_items": eye_items,
        "jaw_pts": jaw_pts,
        "tight_mouth_pts": tight_mouth_pts,
        "pts68": pts68,
        "kps5": kps5,
        "mouth_ref_stats": mouth_ref_stats,
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
    """
    Compute continuous attack/release speaking envelope (0.0 on sustained pauses, 1.0 during speech).
    Fast attack (~60ms) tracks syllable onset crisply; gentle release (~220ms) returns smoothly to the
    portrait's natural resting smile without abrupt mouth snapping.
    """
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    spf = max(1, int(sr / fps))
    rms = np.array([
        np.sqrt(np.mean(audio[i * spf : min((i + 1) * spf, len(audio))] ** 2))
        if i * spf < len(audio) else 0.0
        for i in range(total_frames)
    ], dtype=np.float32)

    # Dual-threshold continuous speech probability
    lo_thr, hi_thr = 0.008, 0.022
    raw = np.clip((rms - lo_thr) / (hi_thr - lo_thr), 0.0, 1.0)

    # Suppress isolated 1-frame noise spikes in silence while keeping brief consonant dips open
    for i in range(1, total_frames - 1):
        if raw[i] < 0.25 and raw[i - 1] > 0.65 and raw[i + 1] > 0.65:
            raw[i] = 0.75

    # Asymmetric attack/release smoothing
    alpha_attack = min(1.0, 1.0 / max(1.0, fps * 0.055))   # ~55ms crisp open
    alpha_release = min(1.0, 1.0 / max(1.0, fps * 0.210))  # ~210ms graceful settle
    env = np.zeros(total_frames, dtype=np.float32)
    cur = float(raw[0]) if total_frames > 0 else 0.0
    for i in range(total_frames):
        target = float(raw[i])
        alpha = alpha_attack if target > cur else alpha_release
        cur += (target - cur) * alpha
        env[i] = cur

    # Clamp very quiet floor cleanly to 0 so pure silence frames skip GFPGAN/blending
    env[env < 0.03] = 0.0
    return np.clip(env, 0.0, 1.0)


# Standard FFHQ 512x512 5-landmark template matching GFPGANv1.4 training alignment
_FFHQ_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


class FastAlignedGFPGAN:
    """
    Zero-detection 1-shot InsightFace-aligned GFPGANv1.4 wrapper.
    Bypasses per-frame RetinaFace detection & face parsing (~2.2x faster on Apple MPS)
    while producing full 512x512 crisp teeth, lip contour, and skin detail matching the source portrait.
    """

    def __init__(self, model_path: str, device: str, weight: float = 0.60) -> None:
        from gfpgan import GFPGANer

        self.device = device
        self.weight = float(weight)
        helper = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=torch.device(device),
        )
        self.net = helper.gfpgan.eval()
        # Free unused detection helper weights immediately to preserve 8GB unified RAM
        del helper

    def restore_face(self, frame_bgr: np.ndarray, kps5_frame: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        m_ffhq, _ = cv2.estimateAffinePartial2D(
            kps5_frame.astype(np.float32), _FFHQ_TEMPLATE_512, method=cv2.LMEDS
        )
        if m_ffhq is None:
            return frame_bgr
        m_inv = cv2.invertAffineTransform(m_ffhq)

        aligned = cv2.warpAffine(
            frame_bgr, m_ffhq, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
        )
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t_in = torch.from_numpy((rgb - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(t_in, return_rgb=False, weight=self.weight)[0]
        out_np = out[0].detach().cpu().float().clamp(-1.0, 1.0).permute(1, 2, 0).numpy()
        out_bgr = cv2.cvtColor(((out_np + 1.0) * 127.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        del t_in, out

        restored = frame_bgr.copy()
        cv2.warpAffine(
            out_bgr, m_inv, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_TRANSPARENT,
            dst=restored,
        )
        return restored


def stabilize_pred_batch(
    pred_np: np.ndarray,
    prev_tail: np.ndarray | None = None,
) -> np.ndarray:
    """
    Energy-adaptive 3-frame temporal stabilization on Wav2Lip 96x96 predictions.
    Eliminates per-frame high-frequency lip-edge jitter on steady vowels while preserving crisp plosives.
    """
    n = len(pred_np)
    if n <= 1:
        return pred_np
    arr = pred_np.astype(np.float32)
    out = arr.copy()
    for i in range(n):
        p_prev = prev_tail if (i == 0 and prev_tail is not None) else (arr[i - 1] if i > 0 else arr[i])
        p_cur = arr[i]
        p_next = arr[i + 1] if i < n - 1 else arr[i]
        # Measure lower-half mouth change to adapt smoothing weight
        delta = float(np.mean(np.abs(p_cur[48:] - p_prev[48:])) + np.mean(np.abs(p_next[48:] - p_cur[48:]))) * 0.5
        # Steady vowels (small delta) get gentle 0.14 neighbor blend; rapid plosives stay razor-crisp (0.03)
        w_nb = float(np.clip(0.15 - (delta - 4.0) * 0.012, 0.03, 0.15))
        out[i] = p_cur * (1.0 - 2.0 * w_nb) + (p_prev + p_next) * w_nb
    return np.clip(out, 0, 255).astype(np.uint8)


def lock_lip_and_skin_color(
    cand_patch: np.ndarray,
    base_patch: np.ndarray,
    outer_lip_local: np.ndarray | None,
    ref_stats: dict | None,
    apply_unsharp: bool = False,
) -> np.ndarray:
    """
    Dual-region LAB statistical color lock:
      - Locks predicted lip tissue (A/B hue+saturation & L luminance) precisely to the source portrait's lipstick.
      - Suppresses Wav2Lip inner-lip purple chromatic artifacts.
      - Keeps teeth clean/white and oral cavity dark without color bleed.
      - Matches surrounding chin/philtrum skin LAB distribution to the original portrait.
    """
    if cand_patch.shape != base_patch.shape or cand_patch.size == 0:
        return cand_patch

    ph, pw = cand_patch.shape[:2]
    lab = cv2.cvtColor(cand_patch, cv2.COLOR_BGR2LAB).astype(np.float32)

    if ref_stats is None or outer_lip_local is None:
        # Fallback to simple lower-half LAB alignment
        b_lab = cv2.cvtColor(base_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        lo = ph // 2
        for c in range(3):
            diff = float(np.mean(b_lab[lo:, :, c]) - np.mean(lab[lo:, :, c]))
            lab[:, :, c] = np.clip(lab[:, :, c] + diff * 0.65, 0.0, 255.0)
        res = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        if apply_unsharp:
            blur = cv2.GaussianBlur(res, (0, 0), sigmaX=1.05)
            res = cv2.addWeighted(res, 1.42, blur, -0.42, 0)
        return res

    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    mouth_cx = float(outer_lip_local[:, 0].mean())
    mouth_cy = float(outer_lip_local[:, 1].mean())
    mw = max(10.0, float(outer_lip_local[:, 0].max() - outer_lip_local[:, 0].min()))
    mh = max(6.0, float(outer_lip_local[:, 1].max() - outer_lip_local[:, 1].min()))

    yy, xx = np.ogrid[:ph, :pw]
    rx, ry = max(8.0, mw * 0.64), max(8.0, mh * 0.98)
    spatial = np.clip(
        1.35 - (((xx - mouth_cx) / rx) ** 2 + ((yy - (mouth_cy + mh * 0.08)) / ry) ** 2),
        0.0,
        1.0,
    ).astype(np.float32)

    skin_A = float(ref_stats["skin_mean"][1])
    # Lip tissue probability: redder than skin, excluding dark oral cavity and bright white teeth
    p_red = np.clip((A - (skin_A + 2.5)) / 9.0, 0.0, 1.0)
    p_not_cavity = np.clip((L - 42.0) / 30.0, 0.0, 1.0)
    p_not_teeth = np.clip(
        1.0 - np.clip((L - 175.0) / 35.0, 0.0, 1.0) * np.clip((skin_A + 6.0 - A) / 6.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    lip_prob = cv2.GaussianBlur(spatial * p_red * p_not_cavity * p_not_teeth, (5, 5), 1.2)

    out_lab = lab.copy()

    # 1. Surrounding skin LAB Reinhard matching
    lo = max(0, int(mouth_cy - mh * 1.4))
    skin_sel = (lip_prob < 0.15) & (L > 65.0)
    skin_sel[:lo, :] = False
    if int(skin_sel.sum()) > 50:
        for c, s_w in [(0, 0.72), (1, 0.88), (2, 0.88)]:
            c_mean = float(lab[:, :, c][skin_sel].mean())
            c_std = max(2.5, float(lab[:, :, c][skin_sel].std()))
            ratio = float(np.clip(ref_stats["skin_std"][c] / c_std, 0.78, 1.28))
            matched = (lab[:, :, c] - c_mean) * ratio + float(ref_stats["skin_mean"][c])
            blend_s = (1.0 - lip_prob) * s_w
            out_lab[:, :, c] = out_lab[:, :, c] * (1.0 - blend_s) + matched * blend_s

    # 2. Lip tissue LAB Reinhard lock to exact portrait lipstick color
    lip_sel = lip_prob > 0.35
    if int(lip_sel.sum()) > 20:
        for c, l_w in [(0, 0.72), (1, 0.96), (2, 0.96)]:
            c_mean = float(lab[:, :, c][lip_sel].mean())
            c_std = max(2.5, float(lab[:, :, c][lip_sel].std()))
            ratio = float(np.clip(ref_stats["lip_std"][c] / c_std, 0.72, 1.32))
            matched = (lab[:, :, c] - c_mean) * ratio + float(ref_stats["lip_mean"][c])
            w_l = lip_prob * l_w
            out_lab[:, :, c] = out_lab[:, :, c] * (1.0 - w_l) + matched * w_l

    # 3. Suppress Wav2Lip inner-lip blue/purple chromatic cast
    purple_mask = (
        spatial
        * np.clip((125.0 - B) / 14.0, 0.0, 1.0)
        * np.clip((L - 25.0) / 40.0, 0.0, 1.0)
    )
    if float(np.max(purple_mask)) > 0.05:
        out_lab[:, :, 2] = (
            out_lab[:, :, 2] * (1.0 - purple_mask * 0.85)
            + float(ref_stats["lip_mean"][2]) * (purple_mask * 0.85)
        )

    matched_bgr = cv2.cvtColor(np.clip(out_lab, 0.0, 255.0).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if apply_unsharp:
        blur = cv2.GaussianBlur(matched_bgr, (0, 0), sigmaX=1.05)
        matched_bgr = cv2.addWeighted(matched_bgr, 1.42, blur, -0.42, 0)
    return matched_bgr


def enhance_mouth_fast(pred_patch: np.ndarray, base_patch: np.ndarray) -> np.ndarray:
    """Backward-compatible helper delegating to lock_lip_and_skin_color with unsharp."""
    return lock_lip_and_skin_color(pred_patch, base_patch, None, None, apply_unsharp=True)


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
    # Prefer tight mouth+chin hull to preserve original cheek & jawline pixels
    blend_pts_src = face_info.get("tight_mouth_pts")
    if blend_pts_src is None:
        blend_pts_src = face_info["jaw_pts"]
    jaw_pts_base = blend_pts_src.reshape(-1, 1, 2) if blend_pts_src is not None else None
    kps5_base = face_info["kps5"].reshape(-1, 1, 2) if face_info.get("kps5") is not None else None
    outer_lip_base = (
        face_info["pts68"][48:60].reshape(-1, 1, 2)
        if face_info.get("pts68") is not None
        else None
    )
    mouth_ref_stats = face_info.get("mouth_ref_stats")

    mel_chunks = compute_mel_chunks(wav2lip_dir, audio_16k, fps)
    total_frames = len(mel_chunks)
    print(f"Video: {w}x{h} @ {fps}fps | {total_frames} frames ({total_frames / fps:.2f}s) | device={device}")

    dx_arr, dy_arr, ang_arr, blink_times = compute_motion_trajectory(audio_16k, total_frames, fps, breathing_amp)
    speak_weights = compute_silence_weights(audio_16k, total_frames, fps)

    # Build text-driven Cantonese Viseme (lip shape) trajectories synced to script/timeline
    try:
        from cantonese_viseme import apply_viseme_mouth_warp, build_viseme_trajectory
        vis_open_arr, vis_width_arr = build_viseme_trajectory(
            total_frames=total_frames,
            fps=fps,
            speak_weights=speak_weights,
            subtitle_timeline=subtitle_timeline,
            subtitle_text=subtitle_text,
            audio_delay_ms=audio_delay_ms,
        )
    except Exception:
        vis_open_arr = np.ones(total_frames, dtype=np.float32)
        vis_width_arr = np.ones(total_frames, dtype=np.float32)
        apply_viseme_mouth_warp = None  # type: ignore[assignment]

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

    fast_gfpgan = None
    if gfpgan_weight > 0:
        if not os.path.exists(gfpgan_weights):
            raise FileNotFoundError(f"Missing GFPGAN weights: {gfpgan_weights}")
        print(f"Loading 1-shot aligned GFPGAN v1.4 on {device} (weight={gfpgan_weight}, zero RetinaFace overhead)")
        fast_gfpgan = FastAlignedGFPGAN(gfpgan_weights, device=device, weight=gfpgan_weight)

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

    blur_k = max(13, int(fw * 0.085) | 1)
    blur_sigma = max(4.5, blur_k / 3.0)

    print("=== [4/4] Streaming Wav2Lip + Viseme + ColorLock + Subtitles -> FFmpeg ===")
    prev_pred_tail: np.ndarray | None = None
    try:
        for b_start in range(0, total_frames, batch_size):
            b_end = min(total_frames, b_start + batch_size)
            b_len = b_end - b_start

            base_frames: list[np.ndarray] = []
            mats_list: list[np.ndarray] = []
            coords_list: list[tuple[int, int, int, int]] = []
            jaw_polys: list[np.ndarray | None] = []
            face_crops_96: list[np.ndarray] = []

            for i in range(b_start, b_end):
                bf, mat = render_base_frame(
                    img, img_f32, face_info, i, fps, breathing_amp,
                    float(dx_arr[i]), float(dy_arr[i]), float(ang_arr[i]), blink_times,
                )
                base_frames.append(bf)
                mats_list.append(mat)

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

            # Temporal lip stabilization across consecutive frames
            pred_np = stabilize_pred_batch(pred_np, prev_tail=prev_pred_tail)
            prev_pred_tail = pred_np[-1].astype(np.float32)

            # Blend each frame & stream straight to FFmpeg
            for j in range(b_len):
                frame_idx = b_start + j
                bf = base_frames[j]
                mat = mats_list[j]
                fy1, fy2, fx1, fx2 = coords_list[j]
                w_speak = float(speak_weights[frame_idx])
                jaw_poly = jaw_polys[j]

                if w_speak <= 0.02:
                    # Pure silence frame: zero blend & skip GFPGAN entirely
                    final_frame = bf
                else:
                    p_up = cv2.resize(pred_np[j], (fx2 - fx1, fy2 - fy1), interpolation=cv2.INTER_CUBIC)
                    outer_lip_local = None
                    if outer_lip_base is not None:
                        ol_t = cv2.transform(outer_lip_base, mat).reshape(-1, 2)
                        outer_lip_local = ol_t - np.array([fx1, fy1], dtype=np.float32)

                    if fast_gfpgan is not None and kps5_base is not None:
                        work_full = bf.copy()
                        work_full[fy1:fy2, fx1:fx2] = p_up
                        kps5_cur = cv2.transform(kps5_base, mat).reshape(-1, 2)
                        restored_full = fast_gfpgan.restore_face(work_full, kps5_cur)
                        cand_patch = restored_full[fy1:fy2, fx1:fx2]
                        enh_patch = lock_lip_and_skin_color(
                            cand_patch,
                            bf[fy1:fy2, fx1:fx2],
                            outer_lip_local,
                            mouth_ref_stats,
                            apply_unsharp=False,
                        )
                    else:
                        enh_patch = lock_lip_and_skin_color(
                            p_up,
                            bf[fy1:fy2, fx1:fx2],
                            outer_lip_local,
                            mouth_ref_stats,
                            apply_unsharp=True,
                        )

                    # Apply text-driven Cantonese Viseme micro-modulation (rounded / spread / open / bilabial)
                    if apply_viseme_mouth_warp is not None and outer_lip_local is not None:
                        mc = (float(outer_lip_local[:, 0].mean()), float(outer_lip_local[:, 1].mean()))
                        mw = float(outer_lip_local[:, 0].max() - outer_lip_local[:, 0].min())
                        mh = float(outer_lip_local[:, 1].max() - outer_lip_local[:, 1].min())
                        enh_patch = apply_viseme_mouth_warp(
                            enh_patch,
                            mouth_center=mc,
                            mouth_w=mw,
                            mouth_h=mh,
                            open_scale=float(vis_open_arr[frame_idx]),
                            width_scale=float(vis_width_arr[frame_idx]),
                        )

                    final_frame = bf.copy()
                    if jaw_poly is not None:
                        # Tight mouth+chin convex hull mask keeps cheeks/nose/jawline 100% original portrait pixels
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
